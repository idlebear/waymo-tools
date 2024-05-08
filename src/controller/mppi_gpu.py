import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

# Need to use the default context instead of using autoinit to initialize a new one
# import pycuda.autoinit
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
from pycuda import characterize

import numpy as np

from matplotlib import pyplot as plt

BLOCK_SIZE = 32


class MPPI:

    # TODO: this is a bit of a hack to make the visibility methods in the GPU code
    #       match the ones in the python code.
    visibility_methods = {
        "Ours": 0,
        "Higgins": 1,
        "Andersen": 2,
        "None": 3,
        "Ignore": 4,
    }

    mod = SourceModule(
        """
    #include <cuda_runtime.h>
    #include <curand.h>
    #include <curand_kernel.h>
    #include <cmath>
    #include <cfloat>

    enum VisibilityMethod {
        OURS = 0,
        HIGGINS = 1,
        ANDERSEN = 2,
        NONE = 3
    };

    struct Costmap_Params {
        int height;
        int width;
        float origin_x;
        float origin_y;
        float resolution;
    };

    struct Optimization_Params {
        int samples;
        float M;
        float dt;
        int num_controls;
        int num_obstacles;
        float x_init[4];
        float u_limits[2];
        float u_dist_limits[2];
        float Q[4];
        float R[2];
        int method;
        float c_lambda;
        float scan_range;
        float discount_factor;
        float vehicle_length;
    };

    struct Object {
        float x;
        float y;
        float radius;
    };

    struct Obstacle {
        Object loc;
        float min_x;
        float min_y;
        float distance;
    };

    struct State {
        float x;
        float y;
        float v;
        float theta;
    };

    struct Control {
        float a;
        float delta;
    };


    __device__
    float obstacle_cost(const Obstacle *obstacles, int num_obstacles, float px, float py, float radius) {
      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = &obstacles[i];
        float dx = obstacle->loc.x - px;
        float dy = obstacle->loc.y - py;
        float d_2 = dx * dx + dy * dy;
        float min_dist = obstacle->loc.radius + radius;

        // printf( "obstacle->loc.x: %f, obstacle->loc.y: %f, obstacle->loc.radius: %f\\n", obstacle->loc.x, obstacle->loc.y, obstacle->loc.radius );
        // printf( "px: %f, py: %f, radius: %f\\n", px, py, radius );
        // printf( "d_2: %f, min_dist: %f\\n", d_2, min_dist * min_dist );

        if (d_2 < min_dist * min_dist) {
            // printf("collision detected\\n");
            // printf( "obstacle->loc.x: %f, obstacle->loc.y: %f, obstacle->loc.radius: %f\\n", obstacle->loc.x, obstacle->loc.y, obstacle->loc.radius );
            // printf( "px: %f, py: %f, radius: %f\\n", px, py, radius );

          return 10000000.0;
        }
      }
      return 0.0;
    }


    __device__
    float higgins_cost(const float M, const Obstacle *obstacles, int num_obstacles, float px, float py, float scan_range) {

      float cost = 0.0;

      float r_fov = scan_range;
      float r_fov_2 = r_fov*r_fov;

      // ( "Checking higgins! px: %f, py: %f, scan_range: %f\\n", px, py, scan_range);

      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = &obstacles[i];
        float dx = obstacle->loc.x - px;
        float dy = obstacle->loc.y - py;
        float d_2 = dx * dx + dy * dy;
        float d = sqrtf(d_2);

        float inner = obstacle->loc.radius / d * (r_fov_2 - d_2);
        auto inner_exp = expf(inner);
        float score;

        // printf( "obstacle->loc.x: %f, obstacle->loc.y: %f, obstacle->loc.radius: %f\\n", obstacle->loc.x, obstacle->loc.y, obstacle->loc.radius );
        // printf( "px: %f, py: %f, scan_range: %f\\n", px, py, scan_range );
        // printf( "d_2: %f, d: %f\\n", d_2, d );
        // printf( "inner: %f, inner_exp: %f\\n", inner, inner_exp );

        if( isinf(inner_exp) || isnan(inner_exp) ) {
          score = inner;
        } else {
          score = logf(1 + inner_exp);
        }
        cost += M * score * score;
      }

      return cost;
    }

    __device__
    float
    andersen_cost( const float M, const Obstacle *obstacles, int num_obstacles, float px, float py, float vx, float vy) {
      float cost = 0.0;
      float v = sqrtf(vx * vx + vy * vy);

      for (int i = 0; i < num_obstacles; i++) {
        auto obstacle = &obstacles[i];
        float dx = obstacle->min_x - px;
        float dy = obstacle->min_y - py;

        // check if the obstacle is in front of the vehicle
        auto dot = dx * vx + dy * vy;
        if (dot > 0) {
          float d = sqrtf(dx * dx + dy * dy);
          // Andersen is a reward
          cost -= M * acosf(dot / (d * v));
          break;   // only consider the closest obstacle
        }
      }

      return cost;
    }


    __device__
    float our_cost(const float M, const float *costmap, int height, int width, float origin_x, float origin_y, float resolution, const float px, const float py,
                   const float discount_factor, const int step) {
      float cost = 0.0;

      auto map_x = int((px - origin_x) / resolution);
      auto map_y = int((py - origin_y) / resolution);

      if (map_x < 0 || map_x >= width || map_y < 0 || map_y >= height) {
        return 10000000.0;
      }

      cost = -M * (costmap[map_y * width + map_x]) * pow(discount_factor, step);

      return cost;
    }


    // Basic step function -- apply the control to advance one step
    __device__
    void euler(const State *state, const Control *control, float vehicle_length, State *result) {
        result->x     = state->v * cosf(state->theta);
        result->y     = state->v * sinf(state->theta);
        result->theta = state->v * tanf(control->delta) / vehicle_length;
        result->v     = control->a;

        // printf( "tan(control->delta): %f\\n", tan(control->delta) );
        // printf( "state->x: %f, state->y: %f, state->v: %f, state->theta: %f\\n", state->x, state->y, state->v, state->theta );
        // printf( "control->a: %f, control->delta: %f\\n", control->a, control->delta );
        // printf( "vehicle_length: %f\\n", vehicle_length );
        // printf( "result->x: %f, result->y: %f, result->v: %f, result->theta: %f\\n", result->x, result->y, result->v, result->theta );

    }

    inline __device__
    void update_state(const State *state, const State *update, float dt, State *result) {
      result->x     = state->x     + update->x * dt;
      result->y     = state->y     + update->y * dt;
      result->v     = state->v     + update->v * dt;
      result->theta = state->theta + update->theta * dt;
    }

    //
    // Also define the Runge-Kutta variant as it is (apparently) a much
    // better approximation of the first order derivative
    //  https://en.wikipedia.org/wiki/Runge-Kutta_methods
    __device__
    void runge_kutta_step(const State *state, const Control *control, float vehicle_length, float dt, State *result) {
      State k1, k2, k3, k4;
      State tmp_state;

      euler(state, control, vehicle_length, &k1);
      update_state(state, &k1, dt / 2, &tmp_state);
      euler(&tmp_state, control, vehicle_length, &k2);
      update_state(state, &k2, dt / 2, &tmp_state);
      euler(&tmp_state, control, vehicle_length, &k3);
      update_state(state, &k3, dt, &tmp_state);
      euler(&tmp_state, control, vehicle_length, &k4);

      result->x = (k1.x + 2 * (k2.x + k3.x) + k4.x) / 6.0;
      result->y = (k1.y + 2 * (k2.y + k3.y) + k4.y) / 6.0;
      result->v = (k1.v + 2 * (k2.v + k3.v) + k4.v) / 6.0;
      result->theta = (k1.theta + 2 * (k2.theta + k3.theta) + k4.theta) / 6.0;
    }


    __device__
    void generate_controls(
            curandState *globalState,
            int index,
            const Control *u_nom,
            const int num_controls,
            const float *u_limits,
            const float *u_dist_limits,
            Control *u_dist
    ) {
      curandState localState = globalState[index];
      for (int i = 0; i < num_controls; i++) {
        float a_dist;
        float delta_dist;
        int count = 0;
        // sample until we get a valid control
        do {
            a_dist = curand_uniform(&localState) * u_dist_limits[0] * 2.0 - u_dist_limits[0];
            count++;
            if (count > 1000) {
                a_dist = 0;
                break;
            }
        } while ((u_nom[i].a + a_dist > u_limits[0]) || (u_nom[i].a + a_dist < -u_limits[0]) );
        count = 0;
        do {
            delta_dist = curand_uniform(&localState) * u_dist_limits[1] * 2.0 - u_dist_limits[1];
            count++;
            if (count > 1000) {
                delta_dist = 0;
                break;
            }
        } while ( (u_nom[i].delta + delta_dist > u_limits[1] ) || ( u_nom[i].delta + delta_dist < -u_limits[1] ) );

        u_dist[i].a = a_dist;
        u_dist[i].delta = delta_dist;
      }
      globalState[index] = localState;
    }


    // External functions -- each is wrapped with extern "C" to prevent name mangling
    // because pycuda doesn't support C++ name mangling
    extern "C" __global__
    void setup_kernel(curandState *state, unsigned long seed) {
      int id = threadIdx.x + blockIdx.x * blockDim.x;
      curand_init(seed, id, 0, &state[id]);
    }


    extern "C" __global__
    void perform_rollout(
            curandState *globalState,
            const float *costmap,
            const Costmap_Params *costmap_args,
            const State *x_nom,   // nominal states, num_controls + 1 x state_size
            const Control *u_nom,   // nominal controls, num_controls x control_size
            const Obstacle *obstacle_data,
            const Optimization_Params *optimization_args,
            Control *u_dists,
            float *u_weights
    ) {
        int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
        int samples = optimization_args->samples;

        for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {

            int height = costmap_args->height;
            int width = costmap_args->width;
            float origin_x = costmap_args->origin_x;
            float origin_y = costmap_args->origin_y;
            float resolution = costmap_args->resolution;

            int num_controls = optimization_args->num_controls;
            int num_obstacles = optimization_args->num_obstacles;
            float M = optimization_args->M;
            float dt = optimization_args->dt;
            const float *u_limits = optimization_args->u_limits;
            const float *u_dist_limits = optimization_args->u_dist_limits;
            const float *Q = optimization_args->Q;
            const float *R = optimization_args->R;
            VisibilityMethod method = (VisibilityMethod) optimization_args->method;

            /*
            printf("samples: %d\\n", samples);
            printf("num_controls: %d\\n", num_controls);
            printf("num_obstacles: %d\\n", num_obstacles);
            printf("M: %f\\n", M);
            printf("dt: %f\\n", dt);
            printf("u_limits: %f, %f\\n", u_limits[0], u_limits[1]);
            printf("Q: %f, %f, %f, %f\\n", Q[0], Q[1], Q[2], Q[3]);
            printf("R: %f, %f\\n", R[0], R[1]);
            printf("method: %d\\n", method);
            printf("SCAN_RANGE: %f\\n", optimization_args->scan_range);
            printf("vehicle_length: %f\\n", optimization_args->vehicle_length);
            printf("DISCOUNT_FACTOR: %f\\n", optimization_args->discount_factor);
            printf("***************\\n");
*/
            float score = 0.0;

            // rollout the trajectory -- assume we are placing the result in the larger u_dist/u_weight arrays
            const State *x_init_state = reinterpret_cast<const State *>(optimization_args->x_init);
            Control *u_dist_controls = reinterpret_cast<Control *>(&u_dists[sample_index * num_controls]);

            generate_controls(globalState, start_sample_index, u_nom, num_controls, u_limits, u_dist_limits, u_dist_controls);

            State current_state;
            State state_step;
            update_state(x_init_state, &state_step, 0, &current_state);
/*
            printf( "initial state.x: %f\\n", current_state.x );
            printf( "initial state.y: %f\\n", current_state.y );
            printf( "initial state.v: %f\\n", current_state.v );
            printf( "initial state.theta: %f\\n", current_state.theta );
*/
            for (int i = 1; i <= num_controls; i++) {
                // generate the next state
                Control c = {u_nom[i - 1].a + u_dist_controls[i - 1].a, u_nom[i - 1].delta + u_dist_controls[i - 1].delta};
                runge_kutta_step(&current_state, &c, optimization_args->vehicle_length, dt, &state_step);
                update_state(&current_state, &state_step, dt, &current_state);
/*
                printf( "x_nom[i].x: %f\\n", x_nom[i].x );
                printf( "x_nom[i].y: %f\\n", x_nom[i].y );
                printf( "x_nom[i].v: %f\\n", x_nom[i].v );
                printf( "x_nom[i].theta: %f\\n", x_nom[i].theta );
                printf( "u_nom[i - 1].a: %f\\n", u_nom[i - 1].a );
                printf( "u_nom[i - 1].delta: %f\\n", u_nom[i - 1].delta );
                printf( "current_state.x: %f\\n", current_state.x );
                printf( "current_state.y: %f\\n", current_state.y );
                printf( "current_state.v: %f\\n", current_state.v );
                printf( "current_state.theta: %f\\n", current_state.theta );
                printf( "c.a: %f\\n", c.a );
                printf( "c.delta: %f\\n", c.delta );
*/

                // penalize error in trajectory
                auto state_err = (x_nom[i].x - current_state.x) * Q[0] *         (x_nom[i].x - current_state.x) +
                                 (x_nom[i].y - current_state.y) * Q[1] *         (x_nom[i].y - current_state.y) +
                                 (x_nom[i].v - current_state.v) * Q[2] *         (x_nom[i].v - current_state.v) +
                                 (x_nom[i].theta - current_state.theta) * Q[3] * (x_nom[i].theta - current_state.theta);

                // penalize control action
                float control_err = (c.a - u_nom[i - 1].a) * R[0] * (c.a - u_nom[i - 1].a) +
                                    (c.delta - u_nom[i - 1].delta) * R[1] * (c.delta - u_nom[i - 1].delta);

                // penalize obstacles
                float obstacle_err = obstacle_cost(obstacle_data, num_obstacles, current_state.x, current_state.y, optimization_args->vehicle_length / 2.0);

                // penalize visibility
                float visibility_err = 0;
                if (method == OURS) {
                    visibility_err = our_cost(M, costmap, height, width, origin_x, origin_y, resolution, current_state.x, current_state.y,
                                              optimization_args->discount_factor, i);
                } else if (method == HIGGINS) {
                    visibility_err = higgins_cost(M, obstacle_data, num_obstacles, current_state.x, current_state.y, optimization_args->scan_range);
                } else if (method == ANDERSEN) {
                    visibility_err = andersen_cost(M, obstacle_data, num_obstacles, current_state.x, current_state.y,
                                                   (x_nom[i].x - x_nom[i-1].x), (x_nom[i].y - x_nom[i-1].y));
                }
                score += state_err + control_err + obstacle_err + visibility_err;
/*
                if( !sample_index && i == num_controls ) {
                    printf( "sample: %d, score: %f\\n", sample_index, score);
                    printf( "   visibility_err: %f\\n", visibility_err );
                    printf( "   obstacle_err: %f\\n", obstacle_err );
                    printf( "   control_err: %f\\n", control_err );
                    printf( "   state_err: %f\\n", state_err );
                }
*/
                if( isnan(score) ){
                    score = 0;
                    break;
                }
            }
            u_weights[sample_index] = score;
        }
    }

    extern "C" __global__
    void calculate_weights(
            int samples,
            float *u_weights,
            float c_lambda,
            float *u_weight_total
    ) {
      int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;

      for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
        u_weights[sample_index] = expf(-u_weights[sample_index] / c_lambda );
//        printf("sample: %d, weight: %f\\n", sample_index, u_weights[sample_index] );
        auto last = atomicAdd(u_weight_total, u_weights[sample_index]);
  //      printf("  normalized weight: %f, total: %f\\n", u_weights[sample_index], last + u_weights[sample_index] );
      }
    }

    //
    // Based on a comment from the following link on checking for zero:
    //
    // https://forums.developer.nvidia.com/t/on-tackling-float-point-precision-issues-in-cuda/79060
    //
    __device__
    inline bool isZero(float f){
        return f >= -FLT_EPSILON && f <= FLT_EPSILON;
    }

    __device__
    inline bool isEqual(float f1, float f2){
        return fabs(f1 - f2) < FLT_EPSILON;
    }

    extern "C" __global__
    void calculate_mppi_control(
            int samples,
            const Control *u_nom,
            Control *u_dist,
            int num_controls,
            const float *u_weights,
            const float *u_weight_total,
            Control *u_mppi
    ) {
      int start_sample_index = blockIdx.x * blockDim.x + threadIdx.x;
      int start_control_index = blockIdx.y * blockDim.y + threadIdx.y;

      for (int sample_index = start_sample_index; sample_index < samples; sample_index += blockDim.x * gridDim.x) {
        for (int control_index = start_control_index; control_index < num_controls; control_index += blockDim.y * gridDim.y) {
            int dist_index = sample_index * num_controls + control_index;
            float weight = u_weights[sample_index]/u_weight_total[0];

            if( sample_index == 0 ) {
                atomicAdd(&(u_mppi[control_index].a), u_nom[control_index].a);
                atomicAdd(&(u_mppi[control_index].delta), u_nom[control_index].delta);
            }
            if( !isEqual( weight, 0 ) ) {
                atomicAdd(&(u_mppi[control_index].a), u_dist[dist_index].a * weight );
                atomicAdd(&(u_mppi[control_index].delta), u_dist[dist_index].delta * weight );
            } else {
                // zero out the dead weights so we can count them
                u_dist[dist_index].a = 0;
                u_dist[dist_index].delta = 0;
            }
        }
      }
    }

    """,
        no_extern_c=True,
    )

    def __init__(
        self,
        vehicle,
        samples,
        seed,
        u_limits,
        u_dist_limits,
        M,
        Q,
        R,
        method,
        c_lambda,
        scan_range,
        discount_factor,
    ):

        self.vehicle = vehicle

        self.samples = np.int32(samples)
        self.c_lambda = np.float32(c_lambda)
        self.method = np.int32(MPPI.visibility_methods[method])
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)

        block = (BLOCK_SIZE, 1, 1)
        grid = (int((self.samples + block[0] - 1) / block[0]), 1)

        # setup the random number generator
        self.globalState_gpu = cuda.mem_alloc(
            block[0] * grid[0] * characterize.sizeof("curandState", "#include <curand_kernel.h>")
        )
        setup_kernel = MPPI.mod.get_function("setup_kernel")
        setup_kernel(
            self.globalState_gpu,
            np.uint32(seed),
            block=block,
            grid=grid,
        )

        self.optimization_dtype = np.dtype(
            [
                ("samples", np.int32),
                ("M", np.float32),
                ("dt", np.float32),
                ("num_controls", np.int32),
                ("num_obstacles", np.int32),
                ("x_init", np.float32, 4),
                ("u_limits", np.float32, 2),
                ("u_dist_limits", np.float32, 2),
                ("Q", np.float32, 4),
                ("R", np.float32, 2),
                ("method", np.int32),
                ("c_lambda", np.float32),
                ("scan_range", np.float32),
                ("discount_factor", np.float32),
                ("vehicle_length", np.float32),
            ]
        )

        self.optimization_args = np.zeros(1, dtype=self.optimization_dtype)
        self.optimization_args["samples"] = np.int32(self.samples)
        self.optimization_args["M"] = np.float32(M)
        self.optimization_args["u_limits"] = np.array(u_limits, dtype=np.float32)
        self.optimization_args["u_dist_limits"] = np.array(u_dist_limits, dtype=np.float32)
        self.optimization_args["Q"] = np.array(Q, dtype=np.float32)
        self.optimization_args["R"] = np.array(R, dtype=np.float32)
        self.optimization_args["method"] = np.int32(MPPI.visibility_methods[method])
        self.optimization_args["c_lambda"] = np.float32(c_lambda)
        self.optimization_args["scan_range"] = np.float32(scan_range)
        self.optimization_args["discount_factor"] = np.float32(discount_factor)
        self.optimization_args["vehicle_length"] = np.float32(vehicle.L)

        self.optimization_args_gpu = cuda.mem_alloc(self.optimization_args.nbytes)

        self.costmap_dtype = np.dtype(
            [
                ("height", np.int32),
                ("width", np.int32),
                ("origin_x", np.float32),
                ("origin_y", np.float32),
                ("resolution", np.float32),
            ]
        )
        self.costmap_args = np.zeros(1, dtype=self.costmap_dtype)
        self.costmap_args_gpu = cuda.mem_alloc(self.costmap_args.nbytes)

    def find_control(self, costmap, origin, resolution, x_init, x_nom, u_nom, actors, dt):
        costmap = costmap.astype(np.float32)
        height, width = costmap.shape
        costmap_gpu = cuda.mem_alloc(costmap.nbytes)
        cuda.memcpy_htod(costmap_gpu, costmap)

        self.costmap_args["height"] = height
        self.costmap_args["width"] = width
        self.costmap_args["origin_x"] = origin[0]
        self.costmap_args["origin_y"] = origin[1]
        self.costmap_args["resolution"] = resolution
        cuda.memcpy_htod(self.costmap_args_gpu, self.costmap_args)

        num_actors = np.int32(len(actors))
        if self.method == MPPI.visibility_methods["Ignore"] or len(actors) == 0:
            actors_gpu = np.intp(0)
        else:
            actors = np.array(actors, dtype=np.float32)
            actors_gpu = cuda.mem_alloc(actors.nbytes)
            cuda.memcpy_htod(actors_gpu, actors)

        u_nom = np.array(u_nom, dtype=np.float32)
        controls_size = u_nom.nbytes
        num_controls, num_control_elements = u_nom.shape
        u_nom_gpu = cuda.mem_alloc(u_nom.nbytes)
        cuda.memcpy_htod(u_nom_gpu, u_nom)

        x_nom = np.array(x_nom, dtype=np.float32)

        x_nom_gpu = cuda.mem_alloc(x_nom.nbytes)
        cuda.memcpy_htod(x_nom_gpu, x_nom)

        # allocate space for the outputs
        u_mppi_gpu = cuda.mem_alloc(controls_size)
        cuda.memset_d8(u_mppi_gpu, 0, controls_size)

        u_weight_gpu = cuda.mem_alloc(int(self.samples * np.float32(1).nbytes))
        u_dist_gpu = cuda.mem_alloc(int(controls_size * self.samples))

        # 1D blocks -- 1 thread per sample
        block = (BLOCK_SIZE, 1, 1)
        grid = (int((self.samples + block[0] - 1) / block[0]), 1)

        # update the optimization parameters
        self.optimization_args["dt"] = np.float32(dt)
        self.optimization_args["num_controls"] = np.int32(num_controls)
        self.optimization_args["num_obstacles"] = np.int32(num_actors)
        self.optimization_args["x_init"] = x_init
        cuda.memcpy_htod(self.optimization_args_gpu, self.optimization_args)

        # # Synchronize the device
        # cuda.Context.synchronize()

        # # perform the rollouts
        func = MPPI.mod.get_function("perform_rollout")
        func(
            self.globalState_gpu,
            costmap_gpu,
            self.costmap_args_gpu,
            x_nom_gpu,
            u_nom_gpu,
            actors_gpu,
            self.optimization_args_gpu,
            u_dist_gpu,
            u_weight_gpu,
            block=block,
            grid=grid,
        )

        # # Synchronize the device
        # cuda.Context.synchronize()

        # calculate the weights -- the block size remains the same
        u_weight_total = cuda.mem_alloc(np.float32(1).nbytes)
        cuda.memset_d8(u_weight_total, 0, np.float32(1).nbytes)

        func = MPPI.mod.get_function("calculate_weights")
        func(
            self.samples,
            u_weight_gpu,
            self.c_lambda,
            u_weight_total,
            block=block,
            grid=grid,
        )

        # # Synchronize the device
        # cuda.Context.synchronize()

        # final evaluation of the control -- 2D blocks, 1 thread per control and sample
        block = (BLOCK_SIZE, BLOCK_SIZE, 1)
        grid = (int((self.samples + block[0] - 1) / block[0]), int((num_controls + block[1] - 1) / block[1]))

        # collect the rollouts into a single control
        func = MPPI.mod.get_function("calculate_mppi_control")
        func(
            self.samples,
            u_nom_gpu,
            u_dist_gpu,
            np.int32(num_controls),
            u_weight_gpu,
            u_weight_total,
            u_mppi_gpu,
            block=block,
            grid=grid,
        )

        # # Synchronize the device
        # cuda.Context.synchronize()

        # copy the results back
        u_dist = np.zeros((self.samples * num_controls * num_control_elements), dtype=np.float32)
        cuda.memcpy_dtoh(u_dist, u_dist_gpu)

        u_mppi = np.zeros_like(u_nom)
        cuda.memcpy_dtoh(u_mppi, u_mppi_gpu)

        # clamp the controls to the limits
        u_mppi[:, 0] = np.clip(
            u_mppi[:, 0], -self.optimization_args["u_limits"][0, 0], self.optimization_args["u_limits"][0, 0]
        )
        u_mppi[:, 1] = np.clip(
            u_mppi[:, 1], -self.optimization_args["u_limits"][0, 1], self.optimization_args["u_limits"][0, 1]
        )

        return u_mppi, u_dist.reshape((self.samples, num_control_elements, -1))


if __name__ == "__main__":

    from time import time
    from controller.ModelParameters.Ackermann import Ackermann4
    from controller.mppi_controller import visualize_variations

    samples = 1
    seed = 123
    u_dist_limits = [2, np.pi / 4]
    u_limits = [4, np.pi / 5]
    M = 0.001
    Q = [1, 1, 1, 1]
    R = [1.0, 1.0]
    method = "Ours"
    c_lambda = 100

    vehicle = Ackermann4()

    tic = time()
    mppi = MPPI(
        vehicle=vehicle,
        samples=samples,
        seed=seed,
        u_limits=u_limits,
        u_dist_limits=u_dist_limits,
        M=M,
        Q=Q,
        R=R,
        method=method,
        c_lambda=c_lambda,
        scan_range=30,
        discount_factor=1,
        vehicle_length=vehicle.L,
    )

    costmap = np.ones((100, 100)) * 0.5
    origin = (0, 0)
    resolution = 1
    x_nom = np.zeros((10, 4))
    x_nom[:, 0] = 1
    x_nom[:, 1] = 0
    x_nom[:, 2] = 1

    u_nom = np.ones((9, 2))
    u_nom[:, 0] = 3.4
    u_nom[:, 1] = 1

    x_init = np.array([0.0, 0.0, 1.0, 0.0])
    actors = [[9, 0, 5, 13, 5, np.sqrt(15 * 15 + 5 * 5)], [20, 20, 5, 25, 5, np.sqrt(25 * 25 + 5 * 5)]]
    actors = np.array(actors, dtype=np.float32)

    dt = 0.1
    u_mppi, u_dist = mppi.find_control(costmap, origin, resolution, x_init, x_nom, u_nom, actors, dt)

    toc = time()
    print(f"Time: {toc - tic}, per sample: {(toc - tic) / samples}")

    # visualize_variations(
    #     figure=1,
    #     vehicle=Ackermann4(),
    #     initial_state=x_init,
    #     u_nom=u_nom,
    #     u_variations=u_dist,
    #     u_weighted=u_mppi,
    #     dt=dt,
    # )

    print(u_mppi)
    print(u_dist)
