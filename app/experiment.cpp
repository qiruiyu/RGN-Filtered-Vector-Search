#include "experiment.hpp"

#include "rgn/rgn.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_set>

namespace rgn::app {
namespace {

bool fileExists(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

void configureNewIndex(
    RgnIndex<float>& index,
    const ProgramOptions& options) {
    index.setEnableQueryRewrite(options.enable_rewrite);
    index.setGuidanceLayerLevel(options.guidance_layer);
    index.setGuidanceLayerRatio(options.guidance_ratio);
    index.setGuidanceLayerMinPerAttributeSet(
        options.representatives_per_attribute_set);
    index.setGuidanceRewriteTopk(options.rewrite_candidates);
}

void validateLoadedIndex(
    RgnIndex<float>& index,
    const Dataset& dataset,
    const ProgramOptions& options) {
    const std::size_t expected_gamma = options.acorn_mode == 1
        ? options.acorn_gamma
        : 1;
    if (index.getCurrentElementCount() != dataset.size() ||
        index.M_ != options.graph_degree ||
        index.getACORNBuildMode() != options.acorn_mode ||
        index.getACORNGamma() != expected_gamma ||
        index.getGuidanceLayerLevel() != options.guidance_layer ||
        index.getGuidanceLayerRatio() != options.guidance_ratio ||
        index.getGuidanceLayerMinPerAttributeSet() !=
            options.representatives_per_attribute_set) {
        throw std::runtime_error(
            "Existing index does not match the dataset or build options; use --rebuild");
    }
    for (std::size_t entity_id = 0; entity_id < dataset.size(); ++entity_id) {
        if (!index.matchesStoredEntity(
                static_cast<labeltype>(entity_id),
                dataset.vector(entity_id),
                dataset.attributes[entity_id])) {
            throw std::runtime_error(
                "Existing index content does not match the dataset; use --rebuild");
        }
    }
    index.setEnableQueryRewrite(options.enable_rewrite);
    index.setGuidanceRewriteTopk(options.rewrite_candidates);
}

double recallAtK(
    const std::vector<std::pair<labeltype, float>>& approximate,
    const QueryRecord& query,
    std::size_t neighbors) {
    std::unordered_set<std::size_t> exact(
        query.exact_neighbors.begin(), query.exact_neighbors.end());
    std::size_t matches = 0;
    for (const auto& result : approximate) {
        if (exact.find(static_cast<std::size_t>(result.first)) != exact.end()) {
            ++matches;
        }
    }
    return static_cast<double>(matches) / static_cast<double>(neighbors);
}

BenchmarkRow benchmarkWidth(
    RgnIndex<float>& index,
    const Dataset& dataset,
    const Workload& workload,
    const ProgramOptions& options,
    std::size_t search_width) {
    index.setEf(search_width);
    double recall_sum = 0.0;
    double nfr_sum = 0.0;
    double distance_sum = 0.0;
    double rewrite_distance_sum = 0.0;
    std::size_t underfilled = 0;
    const auto start = std::chrono::steady_clock::now();

    for (const QueryRecord& query : workload.queries) {
        index.resetSearchDistanceComputations();
        index.resetGuidanceRewriteDistanceComputations();
        index.resetSearchTopCandidatesCount();
        const auto results = index.searchKnn(
            dataset.vector(query.vector_id),
            options.neighbors,
            query.attributes,
            options.search_hops,
            options.rewrite_method,
            options.rewrite_hops,
            options.fusion_strategy,
            options.two_phase_mode);

        for (const auto& result : results) {
            const std::size_t entity_id = static_cast<std::size_t>(result.first);
            if (entity_id >= dataset.size() ||
                !satisfies(dataset.attributes[entity_id], query.attributes)) {
                throw std::runtime_error(
                    "Index returned an entity that violates the query attributes");
            }
        }
        recall_sum += recallAtK(results, query, options.neighbors);
        if (results.size() < options.neighbors) {
            ++underfilled;
        }
        const std::size_t working_size = index.getSearchTopCandidatesCount();
        nfr_sum += 1.0 - static_cast<double>(
            std::min(working_size, search_width)) /
            static_cast<double>(search_width);
        distance_sum += static_cast<double>(
            index.getSearchDistanceComputations());
        rewrite_distance_sum += static_cast<double>(
            index.getGuidanceRewriteDistanceComputations());
    }

    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double query_count = static_cast<double>(workload.queries.size());
    BenchmarkRow row;
    row.search_width = search_width;
    row.recall_at_k = recall_sum / query_count;
    row.queries_per_second = seconds > 0.0 ? query_count / seconds : 0.0;
    row.average_nfr = nfr_sum / query_count;
    row.result_underfill_ratio = static_cast<double>(underfilled) / query_count;
    row.average_distance_computations = distance_sum / query_count;
    row.average_rewrite_distance_computations =
        rewrite_distance_sum / query_count;
    return row;
}

}

ExperimentResult runExperiment(
    const Dataset& dataset,
    const Workload& workload,
    const ProgramOptions& options) {
    if (workload.queries.empty()) {
        throw std::invalid_argument("The generated query workload is empty");
    }

    L2Space space(dataset.dimension);
    std::unique_ptr<RgnIndex<float>> index;
    ExperimentResult result;
    if (!options.rebuild && fileExists(options.index_path)) {
        index = std::make_unique<RgnIndex<float>>(
            &space, options.index_path);
        validateLoadedIndex(*index, dataset, options);
        result.loaded_existing_index = true;
    } else {
        const std::size_t gamma = options.acorn_mode == 1
            ? options.acorn_gamma
            : 1;
        const std::size_t m_beta = options.acorn_mode == 1
            ? options.graph_degree * gamma
            : options.graph_degree;
        index = std::make_unique<RgnIndex<float>>(
            &space,
            dataset.size(),
            options.graph_degree,
            options.construction_width,
            options.index_seed,
            options.acorn_mode,
            gamma,
            m_beta);
        configureNewIndex(*index, options);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t entity_id = 0;
             entity_id < dataset.size();
             ++entity_id) {
            index->addPoint(
                dataset.vector(entity_id),
                static_cast<labeltype>(entity_id),
                dataset.attributes[entity_id]);
        }
        const auto end = std::chrono::steady_clock::now();
        result.construction_seconds =
            std::chrono::duration<double>(end - start).count();
        if (options.save_index) {
            index->saveIndex(options.index_path);
        }
    }

    result.rows.reserve(options.search_widths.size());
    for (std::size_t search_width : options.search_widths) {
        BenchmarkRow row = benchmarkWidth(
            *index, dataset, workload, options, search_width);
        std::cout << "ef=" << row.search_width
                  << " recall=" << std::fixed << std::setprecision(6)
                  << row.recall_at_k
                  << " qps=" << std::setprecision(2)
                  << row.queries_per_second
                  << " nfr=" << std::setprecision(6)
                  << row.average_nfr << '\n';
        result.rows.push_back(row);
    }
    return result;
}

void saveResults(
    const std::string& path,
    const Dataset& dataset,
    const Workload& workload,
    const ProgramOptions& options,
    const ExperimentResult& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open result CSV: " + path);
    }
    output << "ef_search,recall_at_k,qps,average_nfr,result_underfill_ratio,"
              "average_distance_computations,average_rewrite_distance_computations,"
              "average_rho,negative_rho_queries,query_count,query_attribute_count,"
              "entity_count,dimension,M,ef_construction,bias,index_source,"
              "construction_seconds,enable_rewrite,rewrite_method,rewrite_hops,"
              "fusion_strategy,two_phase_mode,search_hops,"
              "guidance_layer,guidance_ratio,representatives,rewrite_candidates,"
              "acorn_mode,acorn_gamma,query_seed,index_seed\n";
    output << std::setprecision(12);
    for (const BenchmarkRow& row : result.rows) {
        output << row.search_width << ','
               << row.recall_at_k << ','
               << row.queries_per_second << ','
               << row.average_nfr << ','
               << row.result_underfill_ratio << ','
               << row.average_distance_computations << ','
               << row.average_rewrite_distance_computations << ','
               << workload.average_correlation << ','
               << workload.negative_correlation_count << ','
               << workload.queries.size() << ','
               << options.query_attribute_count << ','
               << dataset.size() << ','
               << dataset.dimension << ','
               << options.graph_degree << ','
               << options.construction_width << ','
               << options.bias << ','
               << (result.loaded_existing_index ? "loaded" : "built") << ','
               << result.construction_seconds << ','
               << (options.enable_rewrite ? 1 : 0) << ','
               << options.rewrite_method << ','
               << options.rewrite_hops << ','
               << options.fusion_strategy << ','
               << options.two_phase_mode << ','
               << options.search_hops << ','
               << options.guidance_layer << ','
               << options.guidance_ratio << ','
               << options.representatives_per_attribute_set << ','
               << options.rewrite_candidates << ','
               << options.acorn_mode << ','
               << options.acorn_gamma << ','
               << options.query_seed << ','
               << options.index_seed << '\n';
    }
    if (!output) {
        throw std::runtime_error("Failed while writing result CSV: " + path);
    }
}

}
