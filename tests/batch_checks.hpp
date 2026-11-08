#pragma once

#include <batchfit/cpu/batch.hpp>
#include <batchfit/models.hpp>
#include <batchfit/synthetic.hpp>

#include "model_checks.hpp"

#include <atomic>
#include <barrier>
#include <bit>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>

namespace batch_checks {
namespace cpu = batchfit::cpu;
using model_checks::require;

template <typename T> bool same(T a, T b) {
    using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    return std::bit_cast<Bits>(a) == std::bit_cast<Bits>(b);
}
template <typename Function> void rejects(Function function) {
    bool caught = false;
    try {
        function();
    } catch (const std::exception &) {
        caught = true;
    }
    require(caught, "expected batch/pool exception");
}

inline void pool_contract() {
    rejects([] { cpu::ThreadPool invalid(0); });
    std::atomic<std::size_t> total{0};
    {
        cpu::ThreadPool pool(2);
        require(pool.size() == 2, "requested worker count");
        std::vector<std::atomic<unsigned>> visits(1003);
        std::atomic<unsigned> arrivals{0};
        std::barrier gate(2);
        std::mutex ids_mutex;
        std::set<std::thread::id> ids;
        pool.parallel_for(visits.size(), [&](std::size_t begin, std::size_t end) {
            if (arrivals.fetch_add(1) < 2)
                gate.arrive_and_wait();
            {
                std::lock_guard lock(ids_mutex);
                ids.insert(std::this_thread::get_id());
            }
            require(begin < end && end <= visits.size(), "disjoint range bounds");
            for (auto i = begin; i < end; ++i)
                visits[i].fetch_add(1);
        });
        require(ids.size() == 2, "work actually runs concurrently on two jthreads");
        for (const auto &count : visits)
            require(count.load() == 1, "exactly-once coverage including tail");
        pool.parallel_for(
            0, [](std::size_t, std::size_t) { throw std::runtime_error("empty job ran"); });
        rejects([&] { pool.parallel_for(1, {}); });
        bool original_error = false;
        try {
            pool.parallel_for(
                513, [](std::size_t, std::size_t) { throw std::runtime_error("worker failure"); });
        } catch (const std::runtime_error &error) {
            original_error = std::string(error.what()) == "worker failure";
        }
        require(original_error, "worker exception must reach the caller");
        rejects([&] {
            pool.parallel_for(
                1, [&](std::size_t, std::size_t) { pool.parallel_for(1, [](auto, auto) {}); });
        });
        for (unsigned repeat = 0; repeat < 100; ++repeat)
            pool.parallel_for(13, [&](auto begin, auto end) { total.fetch_add(end - begin); });
        require(total == 1300, "pool remains reusable after exceptions and recursive rejection");
    }
    // One compute worker keeps this concurrent-submitter check within the normal budget.
    {
        cpu::ThreadPool serial(1);
        std::exception_ptr caller_failure;
        const auto submit = [&] {
            for (unsigned repeat = 0; repeat < 50; ++repeat)
                serial.parallel_for(17,
                                    [&](auto begin, auto end) { total.fetch_add(end - begin); });
        };
        std::jthread caller([&] {
            try {
                submit();
            } catch (...) {
                caller_failure = std::current_exception();
            }
        });
        submit();
        caller.join();
        if (caller_failure)
            std::rethrow_exception(caller_failure);
        require(total == 3000, "concurrent submissions serialize without losing work");
    }
    for (unsigned repeat = 0; repeat < 10; ++repeat) {
        cpu::ThreadPool transient(2);
        transient.parallel_for(3, [](auto, auto) {});
    }
}

template <typename T, template <typename> class Model>
void equivalence(batchfit::synthetic::Model kind) {
    constexpr unsigned p = Model<T>::parameter_count;
    batchfit::synthetic::Options data;
    data.model = kind;
    data.parameters = batchfit::synthetic::default_ranges(kind);
    data.fits = 257;
    data.points = 65;
    data.chunk_fits = 257;
    data.snr = 40;
    batchfit::synthetic::generate(data, [&](std::span<const float> grid,
                                            const batchfit::synthetic::Chunk &chunk) {
        std::vector<T> x(grid.begin(), grid.end()), y(chunk.y.begin(), chunk.y.end()),
            initial(chunk.true_parameters.size());
        for (std::size_t i = 0; i < initial.size(); ++i)
            initial[i] = T(chunk.true_parameters[i]) * T(1.05) + T(0.01);
        cpu::SolverOptions<T, p> options;
        options.lower[2] = T(0.05);
        if constexpr (p == 4) {
            options.lower[3] = T(0);
            options.upper[3] = T(1);
        }
        if constexpr (p == 8)
            options.lower[5] = T(0.05);
        y[3 * x.size()] = std::numeric_limits<T>::quiet_NaN();
        initial[7 * p] = std::numeric_limits<T>::quiet_NaN();
        cpu::BatchResult<T> a;
        {
            cpu::ThreadPool serial(1);
            a = cpu::fit_batch<T, Model>(x, y, initial, serial, options);
        }
        cpu::ThreadPool parallel(2);
        const auto b = cpu::fit_batch<T, Model>(x, y, initial, parallel, options);
        require(a.status.size() == data.fits && a.parameters.size() == data.fits * p,
                "batch output shapes");
        for (std::size_t i = 0; i < data.fits; ++i) {
            const auto expected =
                cpu::fit_single<T, Model>(x, std::span<const T>(y).subspan(i * x.size(), x.size()),
                                          std::span<const T>(initial).subspan(i * p, p), options);
            require(a.status[i] == expected.status && b.status[i] == expected.status,
                    "per-fit statuses preserved");
            require(a.iterations[i] == expected.iterations &&
                        b.iterations[i] == expected.iterations,
                    "per-fit iterations preserved");
            require(same(a.chi_squared[i], expected.chi_squared) &&
                        same(b.chi_squared[i], expected.chi_squared),
                    "bitwise reference chi squared");
            for (unsigned k = 0; k < p; ++k)
                require(same(a.parameters[i * p + k], expected.parameters[k]) &&
                            same(b.parameters[i * p + k], expected.parameters[k]),
                        "bitwise reference parameters, independent of scheduling");
        }
        require(b.status[3] == batchfit::FitStatus::invalid_input &&
                    b.status[7] == batchfit::FitStatus::invalid_input,
                "bad rows isolated, never dropped");
        const auto empty = cpu::fit_batch<T, Model>(x, {}, {}, parallel, options);
        require(empty.parameters.empty() && empty.status.empty() && empty.iterations.empty() &&
                    empty.chi_squared.empty(),
                "empty batch");
        rejects([&] { cpu::fit_batch<T, Model>({}, y, initial, parallel, options); });
        rejects([&] {
            cpu::fit_batch<T, Model>(x, std::span<const T>(y).first(y.size() - 1), initial,
                                     parallel, options);
        });
        rejects([&] {
            cpu::fit_batch<T, Model>(x, y, std::span<const T>(initial).first(initial.size() - 1),
                                     parallel, options);
        });
        options.max_iterations = 0;
        const auto zero = cpu::fit_batch<T, Model>(x, y, initial, parallel, options);
        require(zero.status[0] == batchfit::FitStatus::max_iterations && zero.iterations[0] == 0,
                "batch iteration budget");
    });
}

template <typename T> void models() {
    equivalence<T, batchfit::models::Gaussian>(batchfit::synthetic::Model::gaussian);
    equivalence<T, batchfit::models::Lorentzian>(batchfit::synthetic::Model::lorentzian);
    equivalence<T, batchfit::models::PseudoVoigt>(batchfit::synthetic::Model::pseudo_voigt);
    equivalence<T, batchfit::models::TwoLorentzian>(batchfit::synthetic::Model::two_lorentzian);
}
} // namespace batch_checks
