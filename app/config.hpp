#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace rgn::app {

struct ProgramOptions {
    std::string data_path;
    std::string index_path{"rgn.index"};
    std::string output_path{"rgn_results.csv"};
    std::size_t neighbors{10};
    std::size_t query_count{100};
    std::size_t query_attribute_count{2};
    std::size_t graph_degree{16};
    std::size_t construction_width{200};
    std::vector<std::size_t> search_widths{50, 100, 200, 400};
    std::uint32_t query_seed{42};
    std::uint32_t index_seed{100};
    double bias{1.0};
    int guidance_layer{1};
    double guidance_ratio{-1.0};
    std::size_t representatives_per_attribute_set{1};
    std::size_t rewrite_candidates{10};
    bool enable_rewrite{true};
    int rewrite_method{2};
    int rewrite_hops{2};
    int fusion_strategy{0};
    int two_phase_mode{1};
    int search_hops{2};
    int acorn_mode{0};
    std::size_t acorn_gamma{2};
    bool rebuild{false};
    bool save_index{true};
    bool dry_run{false};
    bool help{false};
};

ProgramOptions parseOptions(int argc, char** argv);
void validateOptions(const ProgramOptions& options);
void printUsage(std::ostream& output, const char* executable);
void printConfiguration(std::ostream& output, const ProgramOptions& options);

}
