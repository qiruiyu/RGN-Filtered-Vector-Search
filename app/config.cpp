#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace rgn::app {
namespace {

std::string requireValue(int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("Missing value for ") + argv[index]);
    }
    return argv[++index];
}

template<typename T>
T parseUnsigned(const std::string& option, const std::string& value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument(option + " expects a non-negative integer");
    }
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(option + " has an invalid integer value: " + value);
    }
    return static_cast<T>(parsed);
}

int parseInt(const std::string& option, const std::string& value) {
    std::size_t consumed = 0;
    const long parsed = std::stol(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(option + " has an invalid integer value: " + value);
    }
    return static_cast<int>(parsed);
}

double parseDouble(const std::string& option, const std::string& value) {
    std::size_t consumed = 0;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(option + " has an invalid numeric value: " + value);
    }
    return parsed;
}

std::vector<std::size_t> parseWidths(const std::string& value) {
    std::vector<std::size_t> widths;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        widths.push_back(parseUnsigned<std::size_t>("--ef-search", token));
    }
    if (widths.empty()) {
        throw std::invalid_argument("--ef-search must contain at least one value");
    }
    return widths;
}

}

ProgramOptions parseOptions(int argc, char** argv) {
    ProgramOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            options.help = true;
        } else if (option == "--dry-run") {
            options.dry_run = true;
        } else if (option == "--rebuild") {
            options.rebuild = true;
        } else if (option == "--no-save-index") {
            options.save_index = false;
        } else if (option == "--data") {
            options.data_path = requireValue(argc, argv, index);
        } else if (option == "--index") {
            options.index_path = requireValue(argc, argv, index);
        } else if (option == "--output") {
            options.output_path = requireValue(argc, argv, index);
        } else if (option == "--k") {
            options.neighbors = parseUnsigned<std::size_t>(option, requireValue(argc, argv, index));
        } else if (option == "--queries") {
            options.query_count = parseUnsigned<std::size_t>(option, requireValue(argc, argv, index));
        } else if (option == "--query-attributes") {
            options.query_attribute_count = parseUnsigned<std::size_t>(option, requireValue(argc, argv, index));
        } else if (option == "--M") {
            options.graph_degree = parseUnsigned<std::size_t>(option, requireValue(argc, argv, index));
        } else if (option == "--ef-construction") {
            options.construction_width = parseUnsigned<std::size_t>(option, requireValue(argc, argv, index));
        } else if (option == "--ef-search") {
            options.search_widths = parseWidths(requireValue(argc, argv, index));
        } else if (option == "--query-seed") {
            options.query_seed = parseUnsigned<std::uint32_t>(option, requireValue(argc, argv, index));
        } else if (option == "--index-seed") {
            options.index_seed = parseUnsigned<std::uint32_t>(option, requireValue(argc, argv, index));
        } else if (option == "--bias") {
            options.bias = parseDouble(option, requireValue(argc, argv, index));
        } else if (option == "--guidance-layer") {
            options.guidance_layer = parseInt(option, requireValue(argc, argv, index));
        } else if (option == "--guidance-ratio") {
            options.guidance_ratio = parseDouble(option, requireValue(argc, argv, index));
        } else if (option == "--representatives") {
            options.representatives_per_attribute_set = parseUnsigned<std::size_t>(option, requireValue(argc, argv, index));
        } else if (option == "--rewrite-candidates") {
            options.rewrite_candidates = parseUnsigned<std::size_t>(option, requireValue(argc, argv, index));
        } else if (option == "--enable-rewrite") {
            const int value = parseInt(option, requireValue(argc, argv, index));
            if (value != 0 && value != 1) {
                throw std::invalid_argument("--enable-rewrite must be 0 or 1");
            }
            options.enable_rewrite = value == 1;
        } else if (option == "--rewrite-method") {
            options.rewrite_method = parseInt(option, requireValue(argc, argv, index));
        } else if (option == "--rewrite-hops") {
            options.rewrite_hops = parseInt(option, requireValue(argc, argv, index));
        } else if (option == "--fusion") {
            options.fusion_strategy = parseInt(option, requireValue(argc, argv, index));
        } else if (option == "--two-phase") {
            options.two_phase_mode = parseInt(option, requireValue(argc, argv, index));
        } else if (option == "--search-hops") {
            options.search_hops = parseInt(option, requireValue(argc, argv, index));
        } else if (option == "--acorn-mode") {
            options.acorn_mode = parseInt(option, requireValue(argc, argv, index));
        } else if (option == "--acorn-gamma") {
            options.acorn_gamma = parseUnsigned<std::size_t>(option, requireValue(argc, argv, index));
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }
    validateOptions(options);
    return options;
}

void validateOptions(const ProgramOptions& options) {
    if (!options.help && !options.dry_run && options.data_path.empty()) {
        throw std::invalid_argument("--data is required unless --dry-run is used");
    }
    if (options.neighbors == 0 || options.query_count == 0 ||
        options.graph_degree == 0 || options.construction_width == 0) {
        throw std::invalid_argument("k, queries, M, and ef-construction must be positive");
    }
    if (std::any_of(options.search_widths.begin(), options.search_widths.end(),
                    [](std::size_t value) { return value == 0; })) {
        throw std::invalid_argument("all ef-search values must be positive");
    }
    if (options.bias < 0.0 || options.bias > 1.0) {
        throw std::invalid_argument("--bias must be in [0,1]");
    }
    if (options.guidance_layer < 1 ||
        (options.guidance_ratio != -1.0 &&
         (options.guidance_ratio < 0.0 || options.guidance_ratio > 1.0))) {
        throw std::invalid_argument("invalid guidance-layer configuration");
    }
    if (options.rewrite_method < 0 || options.rewrite_method > 2 ||
        options.fusion_strategy < 0 || options.fusion_strategy > 4 ||
        options.two_phase_mode < 0 || options.two_phase_mode > 3) {
        throw std::invalid_argument("invalid rewrite, fusion, or two-phase mode");
    }
    if (options.rewrite_hops < 0 || options.search_hops < 1) {
        throw std::invalid_argument("hop counts are outside their valid ranges");
    }
    if (options.acorn_mode < 0 || options.acorn_mode > 1 ||
        (options.acorn_mode == 1 && options.acorn_gamma < 2)) {
        throw std::invalid_argument("invalid ACORN build configuration");
    }
}

void printUsage(std::ostream& output, const char* executable) {
    output
        << "Usage: " << executable << " --data <dataset.bin> [options]\n"
        << "  --index <path>                 Index path (default: rgn.index)\n"
        << "  --output <path>                CSV path (default: rgn_results.csv)\n"
        << "  --k <n>                        Recall@k (default: 10)\n"
        << "  --queries <n>                  Query count (default: 100)\n"
        << "  --query-attributes <n>         Attributes per query; 0 uses all\n"
        << "  --M <n>                        Graph degree (default: 16)\n"
        << "  --ef-construction <n>          Construction width (default: 200)\n"
        << "  --ef-search <a,b,...>          Search widths (default: 50,100,200,400)\n"
        << "  --query-seed <n>               Query sampling seed (default: 42)\n"
        << "  --index-seed <n>               Graph level seed (default: 100)\n"
        << "  --bias <0..1>                  Qualified-region bias (default: 1)\n"
        << "  --guidance-layer <n>           Shared guidance layer (default: 1)\n"
        << "  --guidance-ratio <-1|0..1>     Layer sampling ratio (default: -1)\n"
        << "  --representatives <n>          Per-attribute-set minimum (default: 1)\n"
        << "  --rewrite-candidates <n>       Mean-rewrite candidates (default: 10)\n"
        << "  --enable-rewrite <0|1>         Disable or enable query rewriting\n"
        << "  --rewrite-method <0..2>        Mean, ORGN, or RGN\n"
        << "  --rewrite-hops <n>             RGN local expansion hops (default: 2)\n"
        << "  --fusion <0..4>                RGN fusion strategy\n"
        << "  --two-phase <0..3>             High/low query-vector mode\n"
        << "  --search-hops <n>              Bottom-layer ACORN hops (default: 2)\n"
        << "  --acorn-mode <0|1>             ACORN-1 or ACORN-gamma build mode\n"
        << "  --acorn-gamma <n>              ACORN-gamma expansion (default: 2)\n"
        << "  --rebuild                      Ignore an existing index\n"
        << "  --no-save-index                Do not persist a rebuilt index\n"
        << "  --dry-run                      Validate and print configuration\n"
        << "  --help                         Show this message\n";
}

void printConfiguration(std::ostream& output, const ProgramOptions& options) {
    output << "RGN configuration\n"
           << "data=" << (options.data_path.empty() ? "<not set>" : options.data_path) << '\n'
           << "index=" << options.index_path << '\n'
           << "output=" << options.output_path << '\n'
           << "M=" << options.graph_degree
           << ", ef_construction=" << options.construction_width
           << ", k=" << options.neighbors
           << ", queries=" << options.query_count
           << ", query_attributes=" << options.query_attribute_count << '\n'
           << "guidance_layer=" << options.guidance_layer
           << ", guidance_ratio=" << options.guidance_ratio
           << ", enable_rewrite=" << options.enable_rewrite
           << ", rewrite_method=" << options.rewrite_method
           << ", rewrite_hops=" << options.rewrite_hops
           << ", fusion=" << options.fusion_strategy
           << ", search_hops=" << options.search_hops
           << ", acorn_mode=" << options.acorn_mode << '\n';
}

}
