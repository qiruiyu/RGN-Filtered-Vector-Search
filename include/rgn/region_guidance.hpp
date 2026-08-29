#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgn {

struct GuidanceCandidate {
    std::uint32_t id{0};
    double distance{0.0};
    std::size_t qualified_degree{0};
    std::size_t distance_rank{0};
    std::size_t connectivity_rank{0};
    const float* vector{nullptr};
};

enum class FusionStrategy : int {
    RankProduct = 0,
    DistanceOnly = 1,
    ConnectivityOnly = 2,
    Random = 3,
    EqualAverage = 4
};

std::vector<float> fuseTransitionVector(
    std::vector<GuidanceCandidate> candidates,
    std::size_t dimension,
    std::size_t result_limit,
    FusionStrategy strategy,
    std::uint32_t random_seed);

}
