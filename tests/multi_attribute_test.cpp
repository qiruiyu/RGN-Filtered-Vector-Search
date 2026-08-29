#include "rgn/rgn.hpp"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testSubsetPredicate() {
    const rgn::AttributeSet entity{3, 1, 2, 1};
    require(entity.values() == std::vector<std::int32_t>({1, 2, 3}),
            "attributes must be sorted and unique");
    require(rgn::satisfies(entity, rgn::AttributeSet{1, 3}),
            "entity {1,2,3} must satisfy query {1,3}");
    require(!rgn::satisfies(rgn::AttributeSet{1, 2}, rgn::AttributeSet{1, 3}),
            "entity {1,2} must not satisfy query {1,3}");
    require(rgn::satisfies(entity, rgn::AttributeSet{}),
            "empty query attributes must match every entity");
}

void testIndexRoundTrip() {
    rgn::L2Space space(2);
    rgn::RgnIndex<float> index(&space, 6, 4, 20, 7);
    index.setGuidanceLayerLevel(1);
    index.setGuidanceLayerMinPerAttributeSet(1);
    index.setGuidanceRewriteTopk(2);

    const std::vector<std::vector<float>> points{
        {0.0f, 0.0f},
        {0.1f, 0.0f},
        {1.0f, 1.0f},
        {1.1f, 1.0f},
        {2.0f, 2.0f},
        {2.1f, 2.0f}};
    const std::vector<rgn::AttributeSet> attributes{
        rgn::AttributeSet{1, 2, 3},
        rgn::AttributeSet{1, 2},
        rgn::AttributeSet{1, 3},
        rgn::AttributeSet{2, 3},
        rgn::AttributeSet{1, 2, 3, 4},
        rgn::AttributeSet{4}};

    for (std::size_t i = 0; i < points.size(); ++i) {
        index.addPoint(points[i].data(), i, attributes[i]);
    }

    bool rejected_duplicate = false;
    try {
        index.addPoint(points.front().data(), 0, attributes.front());
    } catch (const std::runtime_error&) {
        rejected_duplicate = true;
    }
    require(rejected_duplicate,
            "the immutable submission index must reject duplicate labels");

    const float query[2]{0.0f, 0.0f};
    const rgn::AttributeSet query_attributes{1, 3};
    const auto results = index.searchKnn(query, 3, query_attributes, 2, 2, 2, 0, 1);
    require(!results.empty(), "multi-attribute search must return qualified results");
    for (const auto& result : results) {
        require(rgn::satisfies(attributes[result.first], query_attributes),
                "every result must satisfy the query attribute subset");
    }

    const std::string path = "rgn_multi_attribute_test.index";
    index.saveIndex(path);
    rgn::RgnIndex<float> loaded(&space, path);
    const auto loaded_results = loaded.searchKnn(query, 3, query_attributes, 2, 2, 2, 0, 1);
    require(results == loaded_results,
            "save and load must preserve multi-attribute search results");
    std::remove(path.c_str());
}

void testNativeFallbackWithoutRepresentative() {
    rgn::L2Space space(2);
    rgn::RgnIndex<float> index(&space, 4, 4, 20, 11);
    index.setGuidanceLayerLevel(1);
    index.setGuidanceLayerRatio(0.0);
    index.setGuidanceLayerMinPerAttributeSet(0);
    const std::vector<std::vector<float>> points{
        {0.0f, 0.0f}, {0.1f, 0.0f}, {1.0f, 1.0f}, {1.1f, 1.0f}};
    const std::vector<rgn::AttributeSet> attributes{
        rgn::AttributeSet{1, 2}, rgn::AttributeSet{1},
        rgn::AttributeSet{2}, rgn::AttributeSet{1, 2, 3}};
    for (std::size_t index_id = 0; index_id < points.size(); ++index_id) {
        index.addPoint(points[index_id].data(), index_id, attributes[index_id]);
    }
    const float query[2]{0.0f, 0.0f};
    const rgn::AttributeSet query_attributes{1, 2};
    const auto results = index.searchKnn(
        query, 2, query_attributes, 2, 2, 2, 0, 1);
    require(!results.empty(),
            "native fallback must search when no representative exists");
    for (const auto& result : results) {
        require(rgn::satisfies(attributes[result.first], query_attributes),
                "native fallback returned an unqualified entity");
    }
}

}

int main() {
    try {
        testSubsetPredicate();
        testIndexRoundTrip();
        testNativeFallbackWithoutRepresentative();
        std::cout << "multi-attribute tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
