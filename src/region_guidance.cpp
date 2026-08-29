#include "rgn/region_guidance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace rgn {
namespace {

std::vector<std::size_t> rankByDistance(
    const std::vector<GuidanceCandidate>& candidates) {
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        if (candidates[left].distance != candidates[right].distance) {
            return candidates[left].distance < candidates[right].distance;
        }
        return candidates[left].id < candidates[right].id;
    });
    return order;
}

std::vector<std::size_t> rankByConnectivity(
    const std::vector<GuidanceCandidate>& candidates) {
    std::vector<std::size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        if (candidates[left].qualified_degree != candidates[right].qualified_degree) {
            return candidates[left].qualified_degree > candidates[right].qualified_degree;
        }
        return candidates[left].id < candidates[right].id;
    });
    return order;
}

std::vector<std::size_t> denseRanks(
    const std::vector<GuidanceCandidate>& candidates,
    const std::vector<std::size_t>& order,
    bool distance_rank) {
    std::vector<std::size_t> ranks(candidates.size(), 1);
    std::size_t current_rank = 1;
    for (std::size_t position = 0; position < order.size(); ++position) {
        if (position > 0) {
            const GuidanceCandidate& previous = candidates[order[position - 1]];
            const GuidanceCandidate& current = candidates[order[position]];
            const bool changed = distance_rank
                ? current.distance != previous.distance
                : current.qualified_degree != previous.qualified_degree;
            if (changed) {
                current_rank = position + 1;
            }
        }
        ranks[order[position]] = current_rank;
    }
    return ranks;
}

}

std::vector<float> fuseTransitionVector(
    std::vector<GuidanceCandidate> candidates,
    std::size_t dimension,
    std::size_t result_limit,
    FusionStrategy strategy,
    std::uint32_t random_seed) {
    if (dimension == 0) {
        throw std::invalid_argument("transition-vector dimension must be positive");
    }
    if (candidates.empty() || result_limit == 0) {
        return {};
    }
    for (const GuidanceCandidate& candidate : candidates) {
        if (candidate.vector == nullptr || !std::isfinite(candidate.distance)) {
            throw std::invalid_argument("invalid transition-vector candidate");
        }
    }

    std::vector<std::size_t> distance_ranks(candidates.size());
    std::vector<std::size_t> connectivity_ranks(candidates.size());
    const bool has_distance_ranks = std::all_of(
        candidates.begin(), candidates.end(),
        [](const GuidanceCandidate& candidate) { return candidate.distance_rank > 0; });
    const bool has_connectivity_ranks = std::all_of(
        candidates.begin(), candidates.end(),
        [](const GuidanceCandidate& candidate) { return candidate.connectivity_rank > 0; });
    if (has_distance_ranks) {
        std::transform(
            candidates.begin(), candidates.end(), distance_ranks.begin(),
            [](const GuidanceCandidate& candidate) { return candidate.distance_rank; });
    } else {
        distance_ranks = denseRanks(candidates, rankByDistance(candidates), true);
    }
    if (has_connectivity_ranks) {
        std::transform(
            candidates.begin(), candidates.end(), connectivity_ranks.begin(),
            [](const GuidanceCandidate& candidate) { return candidate.connectivity_rank; });
    } else {
        connectivity_ranks = denseRanks(candidates, rankByConnectivity(candidates), false);
    }

    std::vector<std::size_t> selected(candidates.size());
    std::iota(selected.begin(), selected.end(), 0);
    if (strategy == FusionStrategy::Random) {
        std::mt19937 generator(random_seed);
        std::shuffle(selected.begin(), selected.end(), generator);
    } else if (strategy != FusionStrategy::EqualAverage) {
        std::sort(selected.begin(), selected.end(), [&](std::size_t left, std::size_t right) {
            double left_score = 0.0;
            double right_score = 0.0;
            if (strategy == FusionStrategy::RankProduct) {
                left_score = static_cast<double>(distance_ranks[left]) * connectivity_ranks[left];
                right_score = static_cast<double>(distance_ranks[right]) * connectivity_ranks[right];
            } else if (strategy == FusionStrategy::DistanceOnly) {
                left_score = static_cast<double>(distance_ranks[left]);
                right_score = static_cast<double>(distance_ranks[right]);
            } else {
                left_score = static_cast<double>(connectivity_ranks[left]);
                right_score = static_cast<double>(connectivity_ranks[right]);
            }
            if (left_score != right_score) {
                return left_score < right_score;
            }
            return candidates[left].id < candidates[right].id;
        });
    }

    selected.resize(std::min(result_limit, selected.size()));
    std::vector<double> weights(selected.size(), 1.0);
    double weight_sum = 0.0;
    for (std::size_t position = 0; position < selected.size(); ++position) {
        const std::size_t index = selected[position];
        if (strategy == FusionStrategy::RankProduct) {
            weights[position] = 1.0 /
                (static_cast<double>(distance_ranks[index]) * connectivity_ranks[index]);
        } else if (strategy == FusionStrategy::DistanceOnly) {
            weights[position] = 1.0 / static_cast<double>(distance_ranks[index]);
        } else if (strategy == FusionStrategy::ConnectivityOnly) {
            weights[position] = 1.0 / static_cast<double>(connectivity_ranks[index]);
        }
        weight_sum += weights[position];
    }

    if (!(weight_sum > 0.0) || !std::isfinite(weight_sum)) {
        return {};
    }

    std::vector<float> transition(dimension, 0.0f);
    for (std::size_t position = 0; position < selected.size(); ++position) {
        const double normalized_weight = weights[position] / weight_sum;
        const float* vector = candidates[selected[position]].vector;
        for (std::size_t coordinate = 0; coordinate < dimension; ++coordinate) {
            transition[coordinate] += static_cast<float>(normalized_weight * vector[coordinate]);
        }
    }
    return transition;
}

}
