#pragma once

#include "config.hpp"
#include "dataset.hpp"
#include "rgn/attribute_index.hpp"

#include <cstddef>
#include <limits>
#include <vector>

namespace rgn::app {

struct QueryRecord {
    std::size_t vector_id{0};
    AttributeSet attributes;
    std::vector<std::size_t> exact_neighbors;
    std::size_t qualified_count{0};
    double correlation{std::numeric_limits<double>::quiet_NaN()};
};

struct Workload {
    std::vector<QueryRecord> queries;
    double average_correlation{std::numeric_limits<double>::quiet_NaN()};
    std::size_t negative_correlation_count{0};
};

Workload buildWorkload(
    const Dataset& dataset,
    const AttributeIndex& attribute_index,
    const ProgramOptions& options);

}
