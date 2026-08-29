#include "rgn/attribute_index.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>

namespace rgn {

AttributeIndex::AttributeIndex(
    const std::vector<AttributeSet>& entity_attributes)
    : entity_count_(entity_attributes.size()) {
    for (std::size_t entity_id = 0;
         entity_id < entity_attributes.size();
         ++entity_id) {
        for (std::int32_t attribute : entity_attributes[entity_id].values()) {
            postings_[attribute].push_back(entity_id);
        }
    }
}

std::vector<std::size_t> AttributeIndex::qualifiedEntities(
    const AttributeSet& query_attributes) const {
    if (query_attributes.empty()) {
        std::vector<std::size_t> all(entity_count_);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    std::vector<const std::vector<std::size_t>*> posting_lists;
    posting_lists.reserve(query_attributes.size());
    for (std::int32_t attribute : query_attributes.values()) {
        const auto posting = postings_.find(attribute);
        if (posting == postings_.end()) {
            return {};
        }
        posting_lists.push_back(&posting->second);
    }
    std::sort(
        posting_lists.begin(),
        posting_lists.end(),
        [](const auto* left, const auto* right) {
            return left->size() < right->size();
        });

    std::vector<std::size_t> result = *posting_lists.front();
    for (std::size_t index = 1;
         index < posting_lists.size() && !result.empty();
         ++index) {
        std::vector<std::size_t> intersection;
        intersection.reserve(std::min(result.size(), posting_lists[index]->size()));
        std::set_intersection(
            result.begin(),
            result.end(),
            posting_lists[index]->begin(),
            posting_lists[index]->end(),
            std::back_inserter(intersection));
        result.swap(intersection);
    }
    return result;
}

std::size_t AttributeIndex::entityCount() const noexcept {
    return entity_count_;
}

}
