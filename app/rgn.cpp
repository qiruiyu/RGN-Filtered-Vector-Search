#include "config.hpp"
#include "dataset.hpp"
#include "experiment.hpp"
#include "workload.hpp"
#include "rgn/attribute_index.hpp"

#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const rgn::app::ProgramOptions options =
            rgn::app::parseOptions(argc, argv);
        if (options.help) {
            rgn::app::printUsage(std::cout, argv[0]);
            return 0;
        }
        rgn::app::printConfiguration(std::cout, options);
        if (options.dry_run) {
            return 0;
        }

        const rgn::app::Dataset dataset =
            rgn::app::loadDataset(options.data_path);
        const rgn::AttributeIndex attribute_index(dataset.attributes);
        const rgn::app::Workload workload = rgn::app::buildWorkload(
            dataset, attribute_index, options);
        std::cout << "entities=" << dataset.size()
                  << ", dimension=" << dataset.dimension
                  << ", queries=" << workload.queries.size()
                  << ", average_rho=" << std::fixed << std::setprecision(8)
                  << workload.average_correlation
                  << ", negative_rho_queries="
                  << workload.negative_correlation_count << '\n';

        const rgn::app::ExperimentResult result =
            rgn::app::runExperiment(dataset, workload, options);
        rgn::app::saveResults(
            options.output_path, dataset, workload, options, result);
        std::cout << "results=" << options.output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
