# Track and predict the trajectories of pedestrians using the Trajectron++ model.

import numpy as np
import os
import json
import time
import torch
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sb

from trajectron.model.trajectron import Trajectron
from trajectron.environment import Environment, Scene
from trajectron.environment import Environment, Scene, GeometricMap
from trajectron.model.model_registrar import ModelRegistrar
from trajectron.model.online.online_trajectron import OnlineTrajectron

from .agent_track import AgentTrack

standardization = {
    "PEDESTRIAN": {
        "position": {
            "x": {"mean": 0, "std": 1},
            "y": {"mean": 0, "std": 1},
        },
        "velocity": {
            "x": {"mean": 0, "std": 2},
            "y": {"mean": 0, "std": 2},
        },
        "acceleration": {
            "x": {"mean": 0, "std": 1},
            "y": {"mean": 0, "std": 1},
        },
    },
    "VEHICLE": {
        "position": {"x": {"mean": 0, "std": 80}, "y": {"mean": 0, "std": 80}},
        "velocity": {
            "x": {"mean": 0, "std": 15},
            "y": {"mean": 0, "std": 15},
            "norm": {"mean": 0, "std": 15},
        },
        "acceleration": {
            "x": {"mean": 0, "std": 4},
            "y": {"mean": 0, "std": 4},
            "norm": {"mean": 0, "std": 4},
        },
        "heading": {
            "x": {"mean": 0, "std": 1},
            "y": {"mean": 0, "std": 1},
            "°": {"mean": 0, "std": np.pi},
            "d°": {"mean": 0, "std": 1},
        },
    },
    "DELIVERY_BOT": {  # basically people speeds
        "position": {
            "x": {"mean": 0, "std": 1},
            "y": {"mean": 0, "std": 1},
        },
        "velocity": {
            "x": {"mean": 0, "std": 3},
            "y": {"mean": 0, "std": 2},
        },
        "acceleration": {
            "x": {"mean": 0, "std": 1},
            "y": {"mean": 0, "std": 1},
        },
        # TODO: Add heading
    },
}


class Tracker:
    def __init__(
        self,
        initial_timestep,
        args=None,
        scenario_map=None,
        map_origin=None,
        map_pixels_per_meter=1,
        robot=None,
        device=None,
        dt=1.0,
    ) -> None:
        self.scene = Scene(timesteps=initial_timestep + 1, map=map, dt=dt)
        self.timestep = initial_timestep

        if scenario_map is not None:
            if map_origin is None:
                raise ValueError("If scenario_map is provided, origin must also be provided.")

            self.map_origin = map_origin
            self.map_pixels_per_meter = map_pixels_per_meter
            homography = np.eye(3) * map_pixels_per_meter

            self.scenario_map = {}
            self.scenario_map["PEDESTRIAN"] = GeometricMap(
                data=scenario_map["PEDESTRIAN"], homography=homography, description="Pedestrian Spaces"
            )
            self.scenario_map["VEHICLE"] = GeometricMap(
                data=scenario_map["VEHICLE"], homography=homography, description="Vehicle Spaces"
            )

            visualization_map = np.stack(
                (
                    np.maximum([scenario_map["VEHICLE"], scenario_map["PEDESTRIAN"]]),
                    scenario_map["PEDESTRIAN"][1],
                    np.maximum((scenario_map["PEDESTRIAN"]), axis=0),
                ),
                axis=0,
            )
            self.scenario_map["VISUALIZATION"] = GeometricMap(
                data=visualization_map, homography=homography, description="Visualization"
            )

        else:
            self.scenario_map = None

        self.scene.map = self.scenario_map

        self.env = Environment(
            node_type_list=["VEHICLE", "PEDESTRIAN", "DELIVERY_BOT"],
            standardization=standardization,
            scenes=[self.scene],
            robot_type=robot,
        )
        attention_radius = dict()
        attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(self.env.NodeType.PEDESTRIAN, self.env.NodeType.VEHICLE)] = 20.0
        attention_radius[(self.env.NodeType.VEHICLE, self.env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(self.env.NodeType.VEHICLE, self.env.NodeType.VEHICLE)] = 30.0
        self.env.attention_radius = attention_radius

        if not torch.cuda.is_available() or (args.device is not None and args.device == "cpu"):
            args.device = torch.device("cpu")
        else:
            if torch.cuda.device_count() == 1:
                # If you have CUDA_VISIBLE_DEVICES set, which you should,
                # then this will prevent leftover flag arguments from
                # messing with the device allocation.
                args.device = "cuda:0"
            args.device = torch.device(args.device)

        if args.config is None:
            raise ValueError("No configuration supplied!")
        else:
            # Load hyperparameters from json
            config_file = os.path.join(args.model_dir, args.config)
            if not os.path.exists(config_file):
                raise ValueError("Config json not found!")
            with open(config_file, "r") as conf_json:
                self.hyperparams = json.load(conf_json)

        # Add hyperparams from arguments
        self.hyperparams["dynamic_edges"] = args.dynamic_edges
        self.hyperparams["edge_state_combine_method"] = args.edge_state_combine_method
        self.hyperparams["edge_influence_combine_method"] = args.edge_influence_combine_method
        self.hyperparams["edge_addition_filter"] = args.edge_addition_filter
        self.hyperparams["edge_removal_filter"] = args.edge_removal_filter
        self.hyperparams["k_eval"] = args.k_eval
        self.hyperparams["offline_scene_graph"] = args.offline_scene_graph
        self.hyperparams["incl_robot_node"] = args.incl_robot_node
        self.hyperparams["edge_encoding"] = not args.no_edge_encoding
        self.hyperparams["use_map_encoding"] = args.map_encoding

        self.model_registrar = ModelRegistrar(args.model_dir, device)
        self.model_registrar.load_models(args.model_iteration)

        self.trajectron = OnlineTrajectron(
            model_registrar=self.model_registrar, hyperparams=self.hyperparams, device=device
        )
        self.trajectron.set_environment(self.env, initial_timestep)

        self.agent_tracks = {}

        self.samples = args.samples
        self.incremental = args.incremental
        self.dt = dt
        self.history_len = args.history_len

    def step(self, agents, horizon=6, timesteps=1):
        self.timestep += timesteps

        update_agents = []
        for agent in agents:
            agent_id = str(agent["id"])
            update_agents.append(agent_id)
            if agent_id not in self.agent_tracks:
                if agent["type"] == "VEHICLE":
                    agent_type = self.env.NodeType.VEHICLE
                elif agent["type"] == "PEDESTRIAN":
                    agent_type = self.env.NodeType.PEDESTRIAN
                self.agent_tracks[agent_id] = AgentTrack(
                    id=agent_id, agent_type=agent_type, history_length=self.history_len, dt=self.dt
                )
            x, y = agent["position"]
            self.agent_tracks[agent_id].update(
                [
                    [x - self.map_origin[0], y - self.map_origin[1], agent["heading"], self.timestep],
                ]
            )

        start = time.time()
        if self.incremental:
            input_dict = {}
            for agent_id in update_agents:
                track = self.agent_tracks[agent_id]
                input_dict[track.node] = track.get(timestep=self.timestep, state=self.hyperparams["state"])

            dists, preds = self.trajectron.incremental_forward(
                input_dict,
                self.scenario_map,
                prediction_horizon=horizon,
                num_samples=self.samples,
                robot_present_and_future=None,
                full_dist=True,
            )
        else:
            # create new nodes for all agents and build the scene from scratch -- should be the
            # same as the incremental version, but slower
            pass

        end = time.time()
        print(
            f"t={self.timestep}: took {end - start:0.2}s ({1.0/(end - start):0.2} Hz) w/ {len(self.trajectron.nodes)} nodes and {self.trajectron.scene_graph.get_num_edges()} edges"
        )

        for agent_node, agent_prediction in preds.items():
            self.agent_tracks[agent_node.id].set_prediction(self.timestep + 1, agent_prediction)

    def plot_predictions(self, frame, futures=None, results_dir=".", display_offset=0, display_diff=0):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        for agent_node, agent_track in self.agent_tracks.items():
            agent_track.plot_prediction(ax, timestep=frame)

        if futures is not None:
            for future in futures:
                ax.plot(future[:, 0], future[:, 1], linestyle="--", color="black")

        fig_path = os.path.join(results_dir, f"predictions_{frame:05d}.png")
        # ax.set_xlim(display_offset[0], display_offset[0] + display_diff)  # Adjust the x-axis limits
        # ax.set_ylim(display_offset[1], display_offset[1] + display_diff)  # Adjust the y-axis limits

        plt.savefig(fig_path)
        plt.close()
