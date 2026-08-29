#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rgn {

class AttributeSet {
 public:
    AttributeSet() = default;

    AttributeSet(std::initializer_list<std::int32_t> values)
        : values_(values) {
        normalize();
    }

    explicit AttributeSet(std::vector<std::int32_t> values)
        : values_(std::move(values)) {
        normalize();
    }

    const std::vector<std::int32_t>& values() const noexcept {
        return values_;
    }

    std::size_t size() const noexcept {
        return values_.size();
    }

    bool empty() const noexcept {
        return values_.empty();
    }

    bool containsAll(const AttributeSet& required) const noexcept {
        return std::includes(
            values_.begin(),
            values_.end(),
            required.values_.begin(),
            required.values_.end());
    }

    bool operator==(const AttributeSet& other) const noexcept {
        return values_ == other.values_;
    }

    bool operator!=(const AttributeSet& other) const noexcept {
        return !(*this == other);
    }

    bool operator<(const AttributeSet& other) const noexcept {
        return values_ < other.values_;
    }

 private:
    void normalize() {
        std::sort(values_.begin(), values_.end());
        values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
    }

    std::vector<std::int32_t> values_;
};

inline bool satisfies(
    const AttributeSet& entity_attributes,
    const AttributeSet& query_attributes) noexcept {
    return entity_attributes.containsAll(query_attributes);
}

}
