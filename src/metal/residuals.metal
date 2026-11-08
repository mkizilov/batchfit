#include <metal_stdlib>
using namespace metal;

kernel void squared_residuals(device const float *predicted [[buffer(0)]],
                              device const float *observed [[buffer(1)]],
                              device float *result [[buffer(2)]],
                              uint index [[thread_position_in_grid]]) {
    const float residual = predicted[index] - observed[index];
    result[index] = residual * residual;
}
