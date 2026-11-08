#pragma once

#if defined(__METAL_VERSION__)
#define BATCHFIT_NUM_THREAD thread
#else
#define BATCHFIT_NUM_THREAD
#endif

namespace batchfit {
namespace numerics {
// Kept library-free for C++/MSL; fast reassociation can invalidate compensation.
template <typename T>
void add(T term, BATCHFIT_NUM_THREAD T &sum, BATCHFIT_NUM_THREAD T &correction) {
    const T adjusted = term - correction;
    const T next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
}
} // namespace numerics
} // namespace batchfit
#undef BATCHFIT_NUM_THREAD
