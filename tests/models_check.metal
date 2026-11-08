#include <batchfit/models.hpp>
#include <metal_stdlib>

template <typename Model>
void check_model(device const float *input, device float *output, uint index) {
    float params[Model::parameter_count];
    float jacobian[Model::parameter_count];
    for (uint k = 0; k < Model::parameter_count; ++k) {
        params[k] = input[index * 9 + k + 1];
    }
    const float x = input[index * 9];
    Model::jacobian(x, params, jacobian);
    output[index * 9] = Model::eval(x, params);
    for (uint k = 0; k < Model::parameter_count; ++k) {
        output[index * 9 + k + 1] = jacobian[k];
    }
}

kernel void gaussian(device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
                     uint index [[thread_position_in_grid]]) {
    check_model<batchfit::models::Gaussian<float>>(input, output, index);
}
kernel void lorentzian(device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
                       uint index [[thread_position_in_grid]]) {
    check_model<batchfit::models::Lorentzian<float>>(input, output, index);
}
kernel void pseudo_voigt(device const float *input [[buffer(0)]],
                         device float *output [[buffer(1)]],
                         uint index [[thread_position_in_grid]]) {
    check_model<batchfit::models::PseudoVoigt<float>>(input, output, index);
}
kernel void two_lorentzian(device const float *input [[buffer(0)]],
                           device float *output [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
    check_model<batchfit::models::TwoLorentzian<float>>(input, output, index);
}
