#pragma once

#include "attributes.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace rgn {

class AttributeIndex {
 public:
    explicit AttributeIndex(const std::vector<AttributeSet>& entity_attributes);

    std::vector<std::size_t> qualifiedEntities(
        const AttributeSet& query_attributes) const;

    std::size_t entityCount() const noexcept;

 private:
    std::size_t entity_count_{0};
    std::map<std::int32_t, std::vector<std::size_t>> postings_;
};

}
