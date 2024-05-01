import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda

# bad for interoperability
# import pycuda.autoinit
import pycuda.autoprimaryctx

from pycuda.compiler import SourceModule

import numpy as np

BLOCK_SIZE = 32
Y_BLOCK_SIZE = 32
MAX_BLOCKS = 32

mod = SourceModule(
    """

    #include <cmath>
    #include <cfloat>


    // side( v1, v2, point ): checks the position of a point relative to a (directional) infinite line.
    //
    // @param: v1: starting vertex of the line
    // @param: v2: ending vertex, indicating the direction the line is headed
    // @param: point: the point to check
    // @return: +ve if point is left of v1->v2, 0 if on the line, -ve otherwise
    //
    inline __device__ auto
    side(float v1x, float v1y, float v2x, float v2y, float px, float py) -> float {
        return (v2x - v1x) * (py - v1y) - (px - v1x) * (v2y - v1y);
    }

    __device__
    bool test_point(const float *polygon_data, const int num_vertices, float px, float py ) {
        auto winding_number = 0;
        for (int vertex = 0; vertex < num_vertices; vertex++) {
            auto v1x = polygon_data[vertex * 2];
            auto v1y = polygon_data[vertex * 2 + 1];
            auto v2x = polygon_data[((vertex + 1) % num_vertices) * 2];
            auto v2y = polygon_data[((vertex + 1) % num_vertices) * 2 + 1];
            if (v1y <= py) {
                if (v2y > py) {
                    if (side(v1x, v1y, v2x, v2y, px, py) > 0) {
                        winding_number += 1;
                    }
                }
            } else {
                if (v2y <= py) {
                    if (side(v1x, v1y, v2x, v2y, px, py) < 0) {
                        winding_number -= 1;
                    }
                }
            }
        }
        return winding_number != 0 ? true : false;
    }

    __global__
    void check_points(const float *polygon_data, int num_vertices,
                      const float *points_data, int num_points,
                      float *result_data) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (int i = start_index; i < num_points; i += stride) {
            auto px = points_data[i * 2];
            auto py = points_data[i * 2 + 1];

            result_data[i] = test_point( polygon_data, num_vertices, px, py );
        }
    }


    __device__ float
    line_observation( const float* data, int height, int width, int sx, int sy, int ex, int ey ) {
        // Using Bresenham implementation based on description found at:
        //   http://members.chello.at/~easyfilter/Bresenham.pdf
        auto dx = abs(sx-ex);
        auto step_x = sx < ex ? 1 : -1;
        auto dy = -abs(sy-ey);
        auto step_y = sy < ey ? 1 : -1;
        auto error = dx + dy;

        auto observation = 1.0;    // assume the point is initially viewable
        for( ;; ) {
            auto e2 = 2 * error;
            if( e2 >= dy ) {
                if( sx == ex ) {
                    break;
                }
                error = error + dy;
                sx += step_x;
            }
            if( e2 <= dx ) {
                if( sy == ey ) {
                    break;
                }
                error += dx;
                sy += step_y;
            }

            // If we haven't reached the end of the line, apply the view probability to the current observation
            observation *= (1.0 - data[ sy * width + sx]);
            if( observation < FLT_EPSILON*2 ) {          // early stopping condition
                observation = 0;
                break;
            }
        }

        return observation;
    }

    __device__ float
    line_range( const float *polygon_list, const int * polygon_indices, int num_polygons, float sx, float sy, float angle, float max_range, float resolution) {

        float ex, ey;
        auto x_inc = cos(angle) * resolution;
        auto y_inc = sin(angle) * resolution;
        auto dist = 0.0;

        ex = sx;
        ey = sy;
        while( dist < max_range ) {
            ex += x_inc;
            ey += y_inc;
            dist += resolution;

            for( int i = 0; i < num_polygons; i++ ) {
                if( test_point( &polygon_list[ polygon_indices[i] * 2], size_t(polygon_indices[i+1] - polygon_indices[i]), ex, ey ) ) {
                    // in the polygon -- return the current length
                    return( dist );
                }
            }
        }

        // If we reach this point, the ray contacted nothing in flight
        return -1.0;
    }

    __global__ void
    check_visibility(const float *data, const int height, const int width, const int *start,
                          const int *ends, int num_ends, float *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_ends; i += stride) {
            int ex, ey;
            ex = ends[i*2];
            ey = ends[i*2+1];
            results[ey*width + ex] = line_observation( data, height, width, start[0], start[1], ex, ey );
        }
    }

    __global__ void
    check_region_visibility(const float *data, const int height, const int width, const int *starts, int num_starts,
                          const int *ends, int num_ends, float *results ) {

        auto ends_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto ends_stride = blockDim.x * gridDim.x;
        auto starts_index = blockIdx.y * blockDim.y + threadIdx.y;
        auto starts_stride = blockDim.y * gridDim.y;

        for (auto si = starts_index; si < num_starts; si += starts_stride) {
            auto sx = starts[si*2];
            auto sy = starts[si*2+1];

            for (auto ei = ends_index; ei < num_ends; ei += ends_stride) {
                auto ex = ends[ei*2];
                auto ey = ends[ei*2+1];
                results[si * num_ends + ei] = line_observation(data, height, width, sx, sy, ex, ey);
            }
        }
    }

    __global__ void
    faux_ray(const float *polygon_list, const int* polygon_indices, int num_polygons, float start_x, float start_y,
             const float angle_start, const float angle_increment, const int num_rays, const float max_range, const float resolution,
             float *results ) {

        auto start_index = blockIdx.x * blockDim.x + threadIdx.x;
        auto stride = blockDim.x * gridDim.x;

        for (auto i = start_index; i < num_rays; i += stride) {
            auto angle = angle_start + i * angle_increment;
            results[i] = line_range(polygon_list, polygon_indices, num_polygons, start_x, start_y, angle, max_range, resolution );
        }
    }
"""
)


def contains(poly, points):
    num_vertices = len(poly)
    num_points = len(points)

    poly = poly.astype(np.float32)
    points = points.astype(np.float32)

    polygon_size = poly.nbytes
    points_size = points.nbytes
    results_size = points_size // 2

    # allocate device memory for the polygon and the points to test, and copy the data over
    poly_gpu = cuda.mem_alloc(polygon_size)
    cuda.memcpy_htod(poly_gpu, poly)
    points_gpu = cuda.mem_alloc(points_size)
    cuda.memcpy_htod(points_gpu, points)
    results_gpu = cuda.mem_alloc(results_size)

    func = mod.get_function("check_points")
    # block_size = 256
    # num_blocks = max(128, int((num_points + block_size - 1) / block_size))
    block_size = 32
    num_blocks = 32
    func(
        poly_gpu,
        np.int32(num_vertices),
        points_gpu,
        np.int32(num_points),
        results_gpu,
        block=(block_size, num_blocks, 1),
    )

    # copy the results back
    results = np.zeros(
        [
            num_points,
        ],
        dtype=np.float32,
    )
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def visibility(data, start, ends):
    data = data.astype(np.float32)
    height, width = data.shape
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    start = start.astype(np.int32)
    start_gpu = cuda.mem_alloc(start.nbytes)
    cuda.memcpy_htod(start_gpu, start)

    ends = ends.astype(np.int32)
    ends_gpu = cuda.mem_alloc(ends.nbytes)
    cuda.memcpy_htod(ends_gpu, ends)
    num_ends = len(ends)

    results_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memset_d8(results_gpu, 0, data.nbytes)

    func = mod.get_function("check_visibility")
    # block_size = 256
    # num_blocks = max(128, int((num_points + block_size - 1) / block_size))
    block_size = BLOCK_SIZE
    num_blocks = MAX_BLOCKS
    func(
        data_gpu,
        np.int32(height),
        np.int32(width),
        start_gpu,
        ends_gpu,
        np.int32(num_ends),
        results_gpu,
        block=(block_size, num_blocks, 1),
    )

    # copy the results back
    results = np.zeros_like(data, dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def visibility_from_region(data, starts, ends):
    data = data.astype(np.float32)
    height, width = data.shape
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data)

    starts = np.array(starts, dtype=np.int32)
    num_starts = len(starts)
    starts_gpu = cuda.mem_alloc(starts.nbytes)
    cuda.memcpy_htod(starts_gpu, starts)

    ends = np.array(ends, dtype=np.int32)
    ends_gpu = cuda.mem_alloc(ends.nbytes)
    cuda.memcpy_htod(ends_gpu, ends)
    num_ends = len(ends)

    results_size = num_starts * num_ends * np.zeros((1,), dtype=np.float32).nbytes
    results_gpu = cuda.mem_alloc(results_size)

    # block_size = 256
    # num_blocks = max(128, int((num_points + block_size - 1) / block_size))
    block = (BLOCK_SIZE, MAX_BLOCKS, 1)
    grid = (
        max(1, min(MAX_BLOCKS, int((num_ends + BLOCK_SIZE - 1) / BLOCK_SIZE))),
        max(1, min(MAX_BLOCKS, int((num_starts + Y_BLOCK_SIZE - 1) / Y_BLOCK_SIZE))),
    )

    func = mod.get_function("check_region_visibility")
    func(
        data_gpu,
        np.int32(height),
        np.int32(width),
        starts_gpu,
        np.int32(num_starts),
        ends_gpu,
        np.int32(num_ends),
        results_gpu,
        block=block,
        grid=grid,
    )

    # copy the results back
    results = np.zeros((num_starts * num_ends), dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results


def faux_scan(polygons, origin, angle_start, angle_inc, num_rays, max_range, resolution):
    polygons = polygons.astype(np.float32)
    poly_data_size = polygons.nbytes

    if not len(polygons):
        return np.ones((num_rays,), dtype=np.float32) * -1.0

    index = 0
    polygon_indices = [index]
    for poly in polygons:
        index += len(poly)
        polygon_indices.append(index)
    polygon_indices = np.array(polygon_indices, dtype=np.int32)

    scan_size = num_rays * np.zeros((1,), dtype=np.float32).nbytes

    poly_gpu = cuda.mem_alloc(poly_data_size)
    cuda.memcpy_htod(poly_gpu, polygons)

    poly_index_gpu = cuda.mem_alloc(polygon_indices.nbytes)
    cuda.memcpy_htod(poly_index_gpu, polygon_indices)

    results_gpu = cuda.mem_alloc(scan_size)
    cuda.memset_d32(results_gpu, 0, num_rays)

    block = (BLOCK_SIZE, MAX_BLOCKS, 1)

    func = mod.get_function("faux_ray")
    func(
        poly_gpu,
        poly_index_gpu,
        np.int32(len(polygons)),
        np.float32(origin[0]),
        np.float32(origin[1]),
        np.float32(angle_start),
        np.float32(angle_inc),
        np.int32(num_rays),
        np.float32(max_range),
        np.float32(resolution),
        results_gpu,
        block=block,
    )

    # copy the results back
    results = np.zeros(num_rays, dtype=np.float32)
    cuda.memcpy_dtoh(results, results_gpu)

    return results
