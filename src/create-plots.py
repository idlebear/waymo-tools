from enum import IntEnum
import re
import sys
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

# from distinctipy import distinctipy
from os import listdir
from os.path import isfile, join

from latex import write_table

mpl.use("pdf")
# import scienceplots
# plt.style.use(['science', 'ieee'])

# width as measured in inkscape
width = 8  # 3.487
height = width / 1.5


# TABLE formatting
# TITLE = "Two-Sided Occlusions"
# CAPTION = "Cars parked on both sides"
TITLE = "One-Side Occlusions"
CAPTION = "Cars parked on the same side as the AV."


class Tags(IntEnum):
    METHOD = 0
    COST_FN = 1
    STEPS = 2
    HORIZON = 3
    SAMPLES = 4
    SEED = 5


def make_plot(
    x,
    y,
    hue,
    data,
    order,
    colours,
    x_label=None,
    y_label=None,
    legend_location="best",
    plot_name="default.plt",
):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.99, top=0.97)

    sb.lineplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        hue_order=order,
        palette=colours,
        style=hue,
        style_order=order,
        # linewidth=2.5,
    )

    ax.tick_params(axis="both", which="major", labelsize=16)
    handles, labels = ax.get_legend_handles_labels()
    # ax.set_yscale('log')

    ax.legend(
        handles=handles,
        labels=labels,
        loc=legend_location,
        title="Method",
        title_fontsize=18,
        fontsize=16,
    )
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    fig.set_size_inches(width, height)
    fig.savefig(plot_name)
    plt.close(fig)


trajectory_names = [
    "straight",
    "left",
    "slight_left",
    "slight_right",
    "right",
]


def plot_comparison(files):
    sb.set_theme(style="whitegrid")

    plt.rc("font", family="serif", serif="Times")
    plt.rc("text", usetex=True)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.rc("axes", labelsize=12)

    df_list = []

    min_distances = []

    for f in files:
        df = pd.read_csv(f)
        df.fillna("None", inplace=True)
        df["trajectory"] = df["trajectory"].apply(
            lambda x: trajectory_names[int(x)]
        )

        df_list.append(df)

    if len(df_list) == 0:
        print("No data found")
        return

    df = pd.concat(df_list, ignore_index=True, sort=False)


    sb.set_style(style="whitegrid")
    colours = [
        "darkorange",
        # "yellow",
        # "wheat",
        "deepskyblue",
        "royalblue",
        "lightsteelblue",
        # "salmon",
        "silver",
        # "dodgerblue",
        # # 'bisque',
        # "linen",
    ]

    policy_order = trajectory_names

    plot_name = f"trajectory-values-plot.pdf"

    make_plot(
        "step",
        "value",
        hue="trajectory",
        data=df,
        order=trajectory_names,
        colours=colours,
        x_label="Step",
        y_label="Value",
        # legend_location="lower left",
        plot_name=plot_name,
    )


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d",
        "--dir",
        default="recorded_data/",
        type=str,
        help="path to data files",
    )
    argparser.add_argument(
        "-p",
        "--prefix",
        default=None,
        type=str,
        help="subgroup",
    )
    argparser.add_argument(
        "-m",
        "--method",
        default=None,
        type=str,
        help="Select a specific policy to plot",
    )
    argparser.add_argument(
        "-s",
        "--scenario",
        default=None,
        type=str,
        help="Scenario (single-car, straight, etc.)",
    )
    argparser.add_argument(
        "-g",
        "--groupby",
        default=None,
        type=str,
        help="Scenario (single-car, straight, etc.)",
    )

    args = argparser.parse_args()

    if args.prefix is not None:
        prefix_str = args.prefix + "-traj_results_"
    else:
        prefix_str = "traj_results_"

    files = []
    for f in listdir(args.dir):
        if f.startswith(prefix_str):
            if args.scenario is None or args.scenario in f:
                file_path = join(args.dir, f)
                if isfile(file_path):
                    files.append(file_path)

    plot_comparison(files)
