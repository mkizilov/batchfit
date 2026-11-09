#pragma once
namespace batchfit {
enum class UncertaintyStatus {
    available,
    not_converged,
    insufficient_dof,
    singular_matrix,
    active_bound,
    invalid_input
};
}
