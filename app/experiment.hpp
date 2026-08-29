#pragma once

#include "config.hpp"
#include "dataset.hpp"
#include "workload.hpp"

#include <cstddef>
#include <vector>

namespace rgn::app {

struct BenchmarkRow {
    std::size_t search_width{0};
    double recall_at_k{0.0};
    double queries_per_second{0.0};
    double average_nfr{0.0};
    double result_underfill_ratio{0.0};
    double average_distance_computations{0.0};
    double average_rewrite_distance_computations{0.0};
};

struct ExperimentResult {
    bool loaded_existing_index{false};
    double construction_seconds{0.0};
    std::vector<BenchmarkRow> rows;
};

ExperimentResult runExperiment(
    const Dataset& dataset,
    const Workload& workload,
    const ProgramOptions& options);

void saveResults(
    const std::string& path,
    const Dataset& dataset,
    const Workload& workload,
    const ProgramOptions& options,
    const ExperimentResult& result);

}
