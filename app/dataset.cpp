#include "dataset.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rgn::app {
namespace {

#pragma pack(push, 1)
struct DatasetHeader {
    char magic[8];
    std::uint32_t version;
    std::uint64_t entity_count;
    std::uint32_t dimension;
    std::uint32_t attribute_universe_size;
};
#pragma pack(pop)

bool hasMagic(const char* actual, const char* expected) {
    return std::memcmp(actual, expected, 8) == 0;
}

template<typename T>
void readValue(std::istream& input, T& value, const char* field) {
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error(std::string("Failed to read dataset field: ") + field);
    }
}

std::size_t checkedVectorValueCount(
    std::uint64_t entity_count,
    std::uint32_t dimension) {
    if (entity_count == 0 || dimension == 0) {
        throw std::runtime_error("Dataset must contain at least one non-empty vector");
    }
    if (entity_count > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::runtime_error("Dataset vector count overflows addressable memory");
    }
    return static_cast<std::size_t>(entity_count) * dimension;
}

}

std::size_t Dataset::size() const noexcept {
    return attributes.size();
}

const float* Dataset::vector(std::size_t entity_id) const {
    if (entity_id >= size()) {
        throw std::out_of_range("Dataset entity id is out of range");
    }
    return vectors.data() + entity_id * dimension;
}

Dataset loadDataset(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open dataset: " + path);
    }

    DatasetHeader header{};
    readValue(input, header, "header");
    const bool is_rgn_v2 = hasMagic(header.magic, "RGNDATA") && header.version == 2;
    const bool is_legacy_v1 = hasMagic(header.magic, "WIKILBL") && header.version == 1;
    if (!is_rgn_v2 && !is_legacy_v1) {
        throw std::runtime_error(
            "Unsupported dataset format; expected RGNDATA v2 or WIKILBL v1");
    }

    Dataset dataset;
    dataset.dimension = header.dimension;
    dataset.attribute_universe_size = header.attribute_universe_size;
    const std::size_t value_count =
        checkedVectorValueCount(header.entity_count, header.dimension);
    dataset.vectors.resize(value_count);
    dataset.attributes.reserve(static_cast<std::size_t>(header.entity_count));

    for (std::uint64_t entity_id = 0;
         entity_id < header.entity_count;
         ++entity_id) {
        float* destination = dataset.vectors.data() +
            static_cast<std::size_t>(entity_id) * dataset.dimension;
        input.read(
            reinterpret_cast<char*>(destination),
            static_cast<std::streamsize>(dataset.dimension * sizeof(float)));
        if (!input) {
            throw std::runtime_error("Dataset ended while reading vector values");
        }

        if (is_legacy_v1) {
            std::int32_t attribute = 0;
            readValue(input, attribute, "legacy attribute");
            dataset.attributes.emplace_back(
                std::vector<std::int32_t>{attribute});
        } else {
            std::uint32_t attribute_count = 0;
            readValue(input, attribute_count, "attribute count");
            if (attribute_count > 1048576U) {
                throw std::runtime_error("Entity attribute count exceeds the safety limit");
            }
            std::vector<std::int32_t> values(attribute_count);
            if (attribute_count > 0) {
                input.read(
                    reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(
                        attribute_count * sizeof(std::int32_t)));
                if (!input) {
                    throw std::runtime_error(
                        "Dataset ended while reading structured attributes");
                }
            }
            dataset.attributes.emplace_back(std::move(values));
        }
    }

    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Dataset contains trailing bytes");
    }
    return dataset;
}

}
