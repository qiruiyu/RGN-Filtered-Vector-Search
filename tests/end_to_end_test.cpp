#include "../app/dataset.hpp"
#include "../app/experiment.hpp"
#include "../app/workload.hpp"
#include "rgn/attribute_index.hpp"
#include "rgn/rgn.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

#pragma pack(push, 1)
struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint64_t entity_count;
    std::uint32_t dimension;
    std::uint32_t attribute_universe_size;
};
#pragma pack(pop)

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void writeFixture(const std::string& path) {
    const std::vector<std::vector<float>> vectors{
        {0.0f, 0.0f}, {0.1f, 0.0f}, {0.2f, 0.0f},
        {1.0f, 1.0f}, {1.1f, 1.0f}, {1.2f, 1.0f},
        {2.0f, 2.0f}, {2.1f, 2.0f}, {2.2f, 2.0f},
        {3.0f, 3.0f}, {3.1f, 3.0f}, {3.2f, 3.0f}};
    const std::vector<std::vector<std::int32_t>> attributes{
        {1, 2, 5}, {1, 2}, {1, 2, 6},
        {1, 3, 5}, {1, 3}, {1, 3, 6},
        {2, 4, 5}, {2, 4}, {2, 4, 6},
        {3, 4, 5}, {3, 4}, {3, 4, 6}};

    std::ofstream output(path, std::ios::binary);
    Header header{};
    std::memcpy(header.magic, "RGNDATA", 7);
    header.version = 2;
    header.entity_count = vectors.size();
    header.dimension = 2;
    header.attribute_universe_size = 6;
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    for (std::size_t index = 0; index < vectors.size(); ++index) {
        output.write(
            reinterpret_cast<const char*>(vectors[index].data()),
            static_cast<std::streamsize>(vectors[index].size() * sizeof(float)));
        const std::uint32_t count =
            static_cast<std::uint32_t>(attributes[index].size());
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        output.write(
            reinterpret_cast<const char*>(attributes[index].data()),
            static_cast<std::streamsize>(count * sizeof(std::int32_t)));
    }
    require(output.good(), "failed to create the end-to-end fixture");
}

}

int main() {
    const std::string dataset_path = "rgn_end_to_end.data";
    const std::string index_path = "rgn_end_to_end.index";
    const std::string csv_path = "rgn_end_to_end.csv";
    try {
        writeFixture(dataset_path);
        const rgn::app::Dataset dataset = rgn::app::loadDataset(dataset_path);
        require(dataset.size() == 12 && dataset.dimension == 2,
                "RGNDATA v2 dimensions were not preserved");
        const rgn::AttributeIndex attribute_index(dataset.attributes);
        require(attribute_index.qualifiedEntities(rgn::AttributeSet{1, 5}).size() == 2,
                "multi-attribute posting intersection is incorrect");

        rgn::app::ProgramOptions options;
        options.data_path = dataset_path;
        options.index_path = index_path;
        options.output_path = csv_path;
        options.neighbors = 2;
        options.query_count = 6;
        options.query_attribute_count = 2;
        options.graph_degree = 4;
        options.construction_width = 20;
        options.search_widths = {10};
        options.bias = 0.75;
        options.representatives_per_attribute_set = 1;
        options.rewrite_candidates = 2;
        options.rebuild = true;

        rgn::app::ProgramOptions sparse_options = options;
        sparse_options.neighbors = dataset.size() + 1;
        bool rejected_sparse_workload = false;
        try {
            static_cast<void>(rgn::app::buildWorkload(
                dataset, attribute_index, sparse_options));
        } catch (const std::runtime_error&) {
            rejected_sparse_workload = true;
        }
        require(rejected_sparse_workload,
                "workloads with fewer than k qualified entities must be rejected");

        const rgn::app::Workload workload =
            rgn::app::buildWorkload(dataset, attribute_index, options);
        require(workload.queries.size() == 6,
                "query workload size is incorrect");
        for (const auto& query : workload.queries) {
            require(query.attributes.size() == 2,
                    "the workload did not create multi-attribute queries");
        }

        const rgn::app::ExperimentResult built =
            rgn::app::runExperiment(dataset, workload, options);
        require(!built.loaded_existing_index && built.rows.size() == 1,
                "end-to-end build did not produce a benchmark row");
        rgn::app::saveResults(csv_path, dataset, workload, options, built);

        options.rebuild = false;
        const rgn::app::ExperimentResult loaded =
            rgn::app::runExperiment(dataset, workload, options);
        require(loaded.loaded_existing_index,
                "the saved multi-attribute index was not loaded");

        rgn::app::Dataset changed_dataset = dataset;
        changed_dataset.vectors.front() += 1.0f;
        bool rejected_changed_dataset = false;
        try {
            static_cast<void>(rgn::app::runExperiment(
                changed_dataset, workload, options));
        } catch (const std::runtime_error&) {
            rejected_changed_dataset = true;
        }
        require(rejected_changed_dataset,
                "an index must not load for different dataset content");

        bool rejected_dimension_mismatch = false;
        try {
            rgn::L2Space wrong_space(3);
            rgn::RgnIndex<float> wrong_dimension(&wrong_space, index_path);
        } catch (const std::runtime_error&) {
            rejected_dimension_mismatch = true;
        }
        require(rejected_dimension_mismatch,
                "an index must not load with a different vector dimension");

        std::remove(dataset_path.c_str());
        std::remove(index_path.c_str());
        std::remove(csv_path.c_str());
        std::cout << "end-to-end tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::remove(dataset_path.c_str());
        std::remove(index_path.c_str());
        std::remove(csv_path.c_str());
        std::cerr << error.what() << '\n';
        return 1;
    }
}
