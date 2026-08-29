#pragma once

#include "rgn/attributes.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rgn::app {

struct Dataset {
    std::size_t dimension{0};
    std::uint32_t attribute_universe_size{0};
    std::vector<float> vectors;
    std::vector<AttributeSet> attributes;

    std::size_t size() const noexcept;
    const float* vector(std::size_t entity_id) const;
};

Dataset loadDataset(const std::string& path);

}
