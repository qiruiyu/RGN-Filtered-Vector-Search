#include "workload.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace rgn::app {
namespace {

struct AttributeGroup {
    std::size_t count{0};
    std::vector<double> centroid;
};

double squaredDistance(
    const float* left,
    const std::vector<double>& right,
    std::size_t dimension) {
    double distance = 0.0;
    for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
        const double difference = static_cast<double>(left[coordinate]) - right[coordinate];
        distance += difference * difference;
    }
    return distance;
}

double squaredDistance(
    const float* left,
    const float* right,
    std::size_t dimension) {
    double distance = 0.0;
    for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
        const double difference =
            static_cast<double>(left[coordinate]) - right[coordinate];
        distance += difference * difference;
    }
    return distance;
}

std::map<AttributeSet, AttributeGroup> buildGroups(const Dataset& dataset) {
    std::map<AttributeSet, AttributeGroup> groups;
    for (std::size_t entity_id = 0; entity_id < dataset.size(); ++entity_id) {
        AttributeGroup& group = groups[dataset.attributes[entity_id]];
        if (group.centroid.empty()) {
            group.centroid.assign(dataset.dimension, 0.0);
        }
        ++group.count;
        const float* vector = dataset.vector(entity_id);
        for (std::size_t coordinate = 0;
             coordinate < dataset.dimension;
             ++coordinate) {
            group.centroid[coordinate] += vector[coordinate];
        }
    }
    for (auto& entry : groups) {
        for (double& coordinate : entry.second.centroid) {
            coordinate /= static_cast<double>(entry.second.count);
        }
    }
    return groups;
}

AttributeSet chooseQueryAttributes(
    const Dataset& dataset,
    const std::map<AttributeSet, AttributeGroup>& groups,
    std::size_t query_id,
    std::size_t requested_count,
    double bias) {
    const AttributeSet& original = dataset.attributes[query_id];
    if (bias == 0.0 || groups.size() == 1) {
        const auto& values = original.values();
        const std::size_t count = requested_count == 0
            ? values.size()
            : std::min(requested_count, values.size());
        return AttributeSet(std::vector<std::int32_t>(
            values.begin(), values.begin() + count));
    }

    std::vector<std::pair<double, const AttributeSet*>> ranked_groups;
    ranked_groups.reserve(groups.size() - 1);
    const float* query_vector = dataset.vector(query_id);
    for (const auto& entry : groups) {
        if (entry.first == original) {
            continue;
        }
        ranked_groups.emplace_back(
            squaredDistance(query_vector, entry.second.centroid, dataset.dimension),
            &entry.first);
    }
    std::sort(
        ranked_groups.begin(),
        ranked_groups.end(),
        [](const auto& left, const auto& right) {
            if (left.first != right.first) {
                return left.first < right.first;
            }
            return *left.second < *right.second;
        });
    const std::size_t rank = static_cast<std::size_t>(std::llround(
        bias * static_cast<double>(ranked_groups.size())));
    if (rank == 0 || ranked_groups.empty()) {
        const auto& values = original.values();
        const std::size_t count = requested_count == 0
            ? values.size()
            : std::min(requested_count, values.size());
        return AttributeSet(std::vector<std::int32_t>(
            values.begin(), values.begin() + count));
    }
    const auto& values = ranked_groups[
        std::min(rank, ranked_groups.size()) - 1].second->values();
    const std::size_t count = requested_count == 0
        ? values.size()
        : std::min(requested_count, values.size());
    return AttributeSet(std::vector<std::int32_t>(
        values.begin(), values.begin() + count));
}

void populateExactResults(
    QueryRecord& query,
    const Dataset& dataset,
    const AttributeIndex& attribute_index,
    std::size_t neighbors) {
    const std::vector<std::size_t> qualified =
        attribute_index.qualifiedEntities(query.attributes);
    query.qualified_count = qualified.size();
    if (qualified.empty()) {
        return;
    }

    const float* query_vector = dataset.vector(query.vector_id);
    std::vector<double> centroid(dataset.dimension, 0.0);
    for (std::size_t entity_id : qualified) {
        const float* vector = dataset.vector(entity_id);
        for (std::size_t coordinate = 0;
             coordinate < dataset.dimension;
             ++coordinate) {
            centroid[coordinate] += vector[coordinate];
        }
    }
    for (double& coordinate : centroid) {
        coordinate /= static_cast<double>(qualified.size());
    }

    double dispersion = 0.0;
    std::vector<std::pair<double, std::size_t>> distances;
    distances.reserve(qualified.size());
    for (std::size_t entity_id : qualified) {
        const float* vector = dataset.vector(entity_id);
        double centroid_distance = 0.0;
        for (std::size_t coordinate = 0;
             coordinate < dataset.dimension;
             ++coordinate) {
            const double difference =
                static_cast<double>(vector[coordinate]) - centroid[coordinate];
            centroid_distance += difference * difference;
        }
        dispersion += centroid_distance;
        distances.emplace_back(
            squaredDistance(query_vector, vector, dataset.dimension),
            entity_id);
    }
    dispersion /= static_cast<double>(qualified.size());
    const double separation =
        squaredDistance(query_vector, centroid, dataset.dimension);
    const double denominator = dispersion + separation;
    if (denominator > 0.0) {
        query.correlation = (dispersion - separation) / denominator;
    }

    const std::size_t result_count = std::min(neighbors, distances.size());
    std::partial_sort(
        distances.begin(),
        distances.begin() + result_count,
        distances.end(),
        [](const auto& left, const auto& right) {
            if (left.first != right.first) {
                return left.first < right.first;
            }
            return left.second < right.second;
        });
    query.exact_neighbors.reserve(result_count);
    for (std::size_t index = 0; index < result_count; ++index) {
        query.exact_neighbors.push_back(distances[index].second);
    }
}

}

Workload buildWorkload(
    const Dataset& dataset,
    const AttributeIndex& attribute_index,
    const ProgramOptions& options) {
    if (dataset.size() == 0) {
        throw std::invalid_argument("Cannot build a workload from an empty dataset");
    }
    const auto groups = buildGroups(dataset);
    std::vector<std::size_t> entity_ids;
    const bool singleton_attributes = std::all_of(
        dataset.attributes.begin(),
        dataset.attributes.end(),
        [](const AttributeSet& attributes) { return attributes.size() == 1; });
    if (singleton_attributes) {
        std::unordered_map<std::int32_t, std::vector<std::size_t>> groups_by_attribute;
        for (std::size_t entity_id = 0; entity_id < dataset.size(); ++entity_id) {
            groups_by_attribute[dataset.attributes[entity_id].values().front()]
                .push_back(entity_id);
        }
        entity_ids.reserve(dataset.size());
        for (const auto& entry : groups_by_attribute) {
            entity_ids.insert(
                entity_ids.end(), entry.second.begin(), entry.second.end());
        }
    } else {
        entity_ids.resize(dataset.size());
        std::iota(entity_ids.begin(), entity_ids.end(), 0);
    }
    std::mt19937 generator(options.query_seed);
    std::shuffle(entity_ids.begin(), entity_ids.end(), generator);
    Workload workload;
    workload.queries.reserve(std::min(options.query_count, entity_ids.size()));
    double correlation_sum = 0.0;
    std::size_t valid_correlation_count = 0;
    for (std::size_t entity_id : entity_ids) {
        QueryRecord query;
        query.vector_id = entity_id;
        query.attributes = chooseQueryAttributes(
            dataset,
            groups,
            entity_id,
            options.query_attribute_count,
            options.bias);
        populateExactResults(
            query,
            dataset,
            attribute_index,
            options.neighbors);
        if (query.qualified_count < options.neighbors) {
            continue;
        }
        if (std::isfinite(query.correlation)) {
            correlation_sum += query.correlation;
            ++valid_correlation_count;
            if (query.correlation < 0.0) {
                ++workload.negative_correlation_count;
            }
        }
        workload.queries.push_back(std::move(query));
        if (workload.queries.size() == options.query_count) {
            break;
        }
    }
    if (workload.queries.size() != options.query_count) {
        throw std::runtime_error(
            "Not enough query entities have at least k qualified results");
    }
    if (valid_correlation_count > 0) {
        workload.average_correlation =
            correlation_sum / static_cast<double>(valid_correlation_count);
    }
    return workload;
}

}
