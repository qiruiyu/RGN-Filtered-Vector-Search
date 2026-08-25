#include "../../hnswlib/hnswlib.h"
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <limits>
#include <cctype>
#include <cstdio>
#include <cerrno>

// ===================== 新的带标签数据文件头定义 =====================
#pragma pack(push, 1)
struct WikiLabelHeader {
    char magic[8];              // "WIKILBL"
    uint32_t version;           // 1
    uint64_t num_points;        // 向量个数
    uint32_t dim;               // 维度
    uint32_t num_clusters;      // 聚类类别数
};
#pragma pack(pop)

// ===================== 全局缓存 & 分类数据 =====================
std::vector<int> query_indices;
std::unordered_map<hnswlib::labeltype, std::vector<hnswlib::labeltype>> brute_force_cache;

// 从 WIKI_label_xxx.bin 中直接读取出来的类别
std::vector<int> point_categories;

// 运行期根据 point_categories 重新构建
std::unordered_map<int, std::vector<hnswlib::labeltype>> category_to_labels;

// Geometric statistics of the vector distribution for each structured category.
// m2 is the sum of squared Euclidean distances from all category vectors to the mean.
struct CategoryGeometryStats {
    uint64_t count = 0;
    std::vector<double> mean;
    double m2 = 0.0;
};

// Query-level quantities used by the filtered query correlation definition.
struct QueryCorrelationRecord {
    int query_index = -1;
    int original_category = -1;
    int query_category = -1;
    uint64_t qualified_count = 0;
    double sigma2 = 0.0;
    double query_centroid_distance2 = 0.0;
    double rho = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
};

std::vector<CategoryGeometryStats> category_geometry_stats;
std::vector<QueryCorrelationRecord> query_correlation_records;

double brute_force_avg_time = 0.0;
long long total_brute_dist_calc = 0;

// 核心指标：HNSW构建时间(ms)、构建时距离计算次数
double hnsw_build_time = 0.0;
uint64_t hnsw_build_dist_calc = 0;
uint64_t champion_layer_build_dist_calc = 0;
double hnsw_build_champion_layer_ratio = -1.0;
uint32_t hnsw_build_champion_assignment_version = 2;
int hnsw_build_champion_layer_level = 1;
uint64_t hnsw_build_champion_layer_min_per_category = 0;

// 【修改1】heuristic_control 全局变量保留
int heuristic_control = 0;                   

hnswlib::DISTFUNC<float> global_dist_func = nullptr;
void* global_dist_func_param = nullptr;

// 从带标签文件头读取到的全局元信息
uint64_t global_num_points = 0;
uint32_t global_dim = 0;
uint32_t global_num_clusters = 0;

// ===================== 保存/加载 HNSW 构建指标 (去掉冠军列表统计，保留heuristic_control) =====================
void save_build_metrics(const std::string& path) {
    std::cout << "\nSaving HNSW build metrics to: " << path << std::endl;
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open brute.bin for writing");

    ofs.write(reinterpret_cast<const char*>(&hnsw_build_time), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hnsw_build_dist_calc), sizeof(uint64_t));
    // 【修改2】保存 heuristic_control
    ofs.write(reinterpret_cast<const char*>(&heuristic_control), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&champion_layer_build_dist_calc), sizeof(uint64_t));
    ofs.write(reinterpret_cast<const char*>(&hnsw_build_champion_layer_ratio), sizeof(double));
    ofs.write(reinterpret_cast<const char*>(&hnsw_build_champion_assignment_version), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(&hnsw_build_champion_layer_level), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&hnsw_build_champion_layer_min_per_category), sizeof(uint64_t));

    ofs.close();
    std::cout << "Build metrics saved successfully!" << std::endl;
}

void load_build_metrics(const std::string& path) {
    std::cout << "\nLoading HNSW build metrics from: " << path << std::endl;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open brute.bin for reading");

    ifs.read(reinterpret_cast<char*>(&hnsw_build_time), sizeof(double));
    ifs.read(reinterpret_cast<char*>(&hnsw_build_dist_calc), sizeof(uint64_t));
    // 【修改3】加载 heuristic_control
    ifs.read(reinterpret_cast<char*>(&heuristic_control), sizeof(int));

    ifs.read(reinterpret_cast<char*>(&champion_layer_build_dist_calc), sizeof(uint64_t));
    ifs.read(reinterpret_cast<char*>(&hnsw_build_champion_layer_ratio), sizeof(double));
    ifs.read(reinterpret_cast<char*>(&hnsw_build_champion_assignment_version), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(&hnsw_build_champion_layer_level), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&hnsw_build_champion_layer_min_per_category), sizeof(uint64_t));

    if (!ifs) {
        throw std::runtime_error("Failed to read build metrics from brute.bin");
    }

    ifs.close();
    std::cout << "Build metrics loaded successfully! HNSW Build Time: "
              << hnsw_build_time << "ms, Build Dist Calcs: "
              << hnsw_build_dist_calc << ", Heuristic Control: " 
              << heuristic_control << ", Champion Layer Build Dist Calcs: "
              << champion_layer_build_dist_calc << ", Champion Layer Ratio: "
              << hnsw_build_champion_layer_ratio << std::endl;
}

// ===================== 辅助函数 =====================
bool file_exists(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    return file.good();
}

// Publish a fully written temporary file only after the write succeeds. This
// keeps an interrupted index/CSV save from looking like a completed result on
// the next batch run.
void replace_file_from_temporary(
    const std::string& temporary_path,
    const std::string& final_path
) {
    errno = 0;
    if (std::remove(final_path.c_str()) != 0 && errno != ENOENT) {
        throw std::runtime_error(
            "Failed to replace existing output file " + final_path +
            ": " + std::strerror(errno));
    }

    errno = 0;
    if (std::rename(temporary_path.c_str(), final_path.c_str()) != 0) {
        throw std::runtime_error(
            "Failed to publish temporary output " + temporary_path +
            " as " + final_path + ": " + std::strerror(errno));
    }
}

bool build_metrics_match_champion_layer_ratio(
    const std::string& filename,
    int expected_heuristic_control,
    double expected_ratio,
    uint32_t expected_assignment_version,
    int expected_layer_level,
    size_t expected_min_per_category
) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    double saved_build_time = 0.0;
    uint64_t saved_build_dist_calc = 0;
    int saved_heuristic_control = 0;
    uint64_t saved_champion_build_dist_calc = 0;
    double saved_champion_layer_ratio = -1.0;
    uint32_t saved_champion_assignment_version = 0;
    int saved_champion_layer_level = 0;
    uint64_t saved_champion_layer_min_per_category = 0;
    file.read(reinterpret_cast<char*>(&saved_build_time), sizeof(double));
    file.read(reinterpret_cast<char*>(&saved_build_dist_calc), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&saved_heuristic_control), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_champion_build_dist_calc), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&saved_champion_layer_ratio), sizeof(double));
    file.read(reinterpret_cast<char*>(&saved_champion_assignment_version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&saved_champion_layer_level), sizeof(int));
    file.read(reinterpret_cast<char*>(&saved_champion_layer_min_per_category), sizeof(uint64_t));
    return file.good() &&
        saved_heuristic_control == expected_heuristic_control &&
        saved_champion_layer_ratio == expected_ratio &&
        saved_champion_assignment_version == expected_assignment_version &&
        saved_champion_layer_level == expected_layer_level &&
        saved_champion_layer_min_per_category ==
            static_cast<uint64_t>(expected_min_per_category);
}

// ===================== ✅ 核心修改：基于BIAS_DEGREE的偏差类别计算 =====================
int get_biased_category(int original_cat, float bias_degree) {
    if (global_num_clusters == 0) {
        throw std::runtime_error("global_num_clusters is 0, cannot compute biased category.");
    }
    if (original_cat < 0 || original_cat >= static_cast<int>(global_num_clusters)) {
        throw std::runtime_error("Invalid original category for biased computation.");
    }
    if (bias_degree < 0.0f || bias_degree > 1.0f) {
        throw std::runtime_error("BIAS_DEGREE must be in [0, 1]!");
    }

    const int c = static_cast<int>(global_num_clusters);
    const float half_c = c / 2.0f;
    int far_cat;

    if (original_cat > half_c) {
        far_cat = c - 1;
    } else {
        far_cat = 0;
    }

    float biased_cat_float = original_cat + bias_degree * (far_cat - original_cat);
    
    int biased_cat = static_cast<int>(std::round(biased_cat_float));
    biased_cat = std::max(0, std::min(c - 1, biased_cat));

    return biased_cat;
}

// ===================== 读取新的 WIKI_label_xxx.bin =====================
std::vector<float> load_labeled_data(
    const std::string& data_path
) {
    std::cout << "========================================" << std::endl;
    std::cout << "Step 1: Loading Labeled Dataset" << std::endl;
    std::cout << "========================================" << std::endl;

    std::ifstream ifs(data_path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open labeled data file: " + data_path);
    }

    WikiLabelHeader header{};
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!ifs) {
        throw std::runtime_error("Failed to read labeled data header.");
    }

    const char expected_magic[8] = {'W','I','K','I','L','B','L','\0'};
    if (std::memcmp(header.magic, expected_magic, 7) != 0) {
        throw std::runtime_error("Invalid file magic. This file is not a valid WIKI_label.bin");
    }

    if (header.version != 1) {
        throw std::runtime_error("Unsupported WIKI_label.bin version: " + std::to_string(header.version));
    }

    global_num_points = header.num_points;
    global_dim = header.dim;
    global_num_clusters = header.num_clusters;

    if (global_num_points == 0 || global_dim == 0 || global_num_clusters == 0) {
        throw std::runtime_error("Invalid header values in WIKI_label.bin");
    }

    std::vector<float> data_buffer(static_cast<size_t>(global_num_points) * global_dim);
    point_categories.resize(static_cast<size_t>(global_num_points));

    std::vector<float> tmp_vec(global_dim);
    int32_t tmp_label = 0;

    for (uint64_t i = 0; i < global_num_points; ++i) {
        ifs.read(reinterpret_cast<char*>(tmp_vec.data()), global_dim * sizeof(float));
        ifs.read(reinterpret_cast<char*>(&tmp_label), sizeof(int32_t));

        if (!ifs) {
            throw std::runtime_error("Data file read error (corrupted or incomplete labeled file)");
        }

        std::copy(tmp_vec.begin(), tmp_vec.end(), data_buffer.begin() + static_cast<size_t>(i) * global_dim);
        point_categories[static_cast<size_t>(i)] = static_cast<int>(tmp_label);
    }

    ifs.close();

    std::cout << " Labeled dataset loaded successfully" << std::endl;
    std::cout << " Num Points   : " << global_num_points << std::endl;
    std::cout << " Dimension    : " << global_dim << std::endl;
    std::cout << " Num Clusters : " << global_num_clusters << std::endl;

    return data_buffer;
}

// ===================== 根据 point_categories 重建类别分组 =====================
void preprocess_category_groups(uint64_t max_elements) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Step 2: Preprocessing Category Groups (From Labeled File)" << std::endl;
    std::cout << "========================================" << std::endl;

    category_to_labels.clear();

    for (uint64_t i = 0; i < max_elements; ++i) {
        int cat = point_categories[static_cast<size_t>(i)];
        category_to_labels[cat].push_back(static_cast<hnswlib::labeltype>(i));
    }

    std::cout << " Successfully grouped vectors by category" << std::endl;
    for (const auto& pair : category_to_labels) {
        std::cout << " Category " << pair.first << ": " << pair.second.size() << " vectors" << std::endl;
    }
}

// ===================== 构建 HNSW (去掉冠军列表构建统计) =====================
double build_hnsw_index(
    hnswlib::HierarchicalNSW<float>* alg_hnsw,
    uint64_t max_elements,
    int dim,
    const std::vector<float>& data_buffer
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Step 3: Building HNSW Index (Using Categories From Labeled File)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Configuration: M=" << alg_hnsw->M_ << ", ef_construction=" << alg_hnsw->ef_construction_ << std::endl;
    std::cout << "Inserting " << max_elements << " points with file-based categories into HNSW..." << std::endl;

    alg_hnsw->resetBuildDistanceComputations();
    alg_hnsw->resetChampionLayerBuildDistanceComputations();
    auto build_start = std::chrono::steady_clock::now();

    for (uint64_t i = 0; i < max_elements; ++i) {
        if (i % 10000 == 0) {
            std::cout << "\rProgress: " << i << " / " << max_elements << " points inserted" << std::flush;
        }
        const float* vec = data_buffer.data() + static_cast<size_t>(i) * dim;
        alg_hnsw->addPoint(vec, static_cast<hnswlib::labeltype>(i), point_categories[static_cast<size_t>(i)]);
    }
    std::cout << "\rProgress: " << max_elements << " / " << max_elements << " points inserted" << std::flush;

    auto build_end = std::chrono::steady_clock::now();
    hnsw_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    hnsw_build_dist_calc = alg_hnsw->getBuildDistanceComputations();
    champion_layer_build_dist_calc =
        alg_hnsw->getChampionLayerBuildDistanceComputations();
    hnsw_build_champion_layer_ratio = alg_hnsw->getChampionLayerRatio();
    hnsw_build_champion_assignment_version =
        alg_hnsw->getChampionLayerAssignmentVersion();
    hnsw_build_champion_layer_level = alg_hnsw->getChampionLayerLevel();
    hnsw_build_champion_layer_min_per_category =
        static_cast<uint64_t>(alg_hnsw->getChampionLayerMinPerCategory());

    std::cout << "\n HNSW index built successfully (time: " << hnsw_build_time << "ms)" << std::endl;
    std::cout << " Build distance calculations: " << hnsw_build_dist_calc << std::endl;
    std::cout << " Champion layer build distance calculations: "
              << champion_layer_build_dist_calc << std::endl;
    return hnsw_build_time;
}

// ===================== 分析所有层的拓扑结构 (含有效边比例) =====================
void analyze_all_layers_topology(
    hnswlib::HierarchicalNSW<float>* alg_hnsw,
    std::vector<uint64_t>& out_hetero_edges_per_layer,
    std::vector<uint64_t>& out_homo_edges_per_layer,
    std::vector<double>& out_hetero_ratio_per_layer,
    std::vector<double>& out_effective_edge_ratio_per_layer
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Step 4: Analyzing Topology of All Layers" << std::endl;
    std::cout << "========================================" << std::endl;

    out_hetero_edges_per_layer.clear();
    out_homo_edges_per_layer.clear();
    out_hetero_ratio_per_layer.clear();
    out_effective_edge_ratio_per_layer.clear();

    int max_level = alg_hnsw->getMaxLevel();
    int M = alg_hnsw->M_;
    std::cout << "Index Max Level: " << max_level << " (Analyzing layers 0 to " << max_level << ")" << std::endl;
    std::cout << "Using M = " << M << " for effective edge ratio calculation." << std::endl;

    uint64_t total_hetero_all = 0;
    uint64_t total_homo_all = 0;

    for (int l = 0; l <= max_level; ++l) {
        uint64_t hetero = 0;
        uint64_t homo = 0;
        
        double ratio = alg_hnsw->getHeterogeneousEdgeRatioAtLayer(l, hetero, homo);
        
        out_hetero_edges_per_layer.push_back(hetero);
        out_homo_edges_per_layer.push_back(homo);
        out_hetero_ratio_per_layer.push_back(ratio);

        total_hetero_all += hetero;
        total_homo_all += homo;

        uint64_t actual_edges = hetero + homo;
        double m_power = std::pow(M, l);
        uint64_t theoretical_nodes = static_cast<uint64_t>(std::round(global_num_points / m_power));
        
        if (theoretical_nodes == 0) theoretical_nodes = 1;
        
        uint64_t max_possible_edges = theoretical_nodes * static_cast<uint64_t>(M);
        double effective_ratio = 0.0;
        
        if (max_possible_edges > 0) {
            effective_ratio = static_cast<double>(actual_edges) / static_cast<double>(max_possible_edges);
        }
        
        out_effective_edge_ratio_per_layer.push_back(effective_ratio);
    }

    std::cout << "\nLayer-wise Topology Analysis:" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(10) << "Layer" 
              << std::right << std::setw(20) << "Hetero Edges" 
              << std::right << std::setw(20) << "Homo Edges" 
              << std::right << std::setw(20) << "Total Edges"
              << std::right << std::setw(15) << "Hetero Ratio"
              << std::right << std::setw(20) << "Effective Ratio" << std::endl;
    std::cout << "------------------------------------------------------------------------------------------------------------------" << std::endl;

    for (int l = 0; l <= max_level; ++l) {
        std::cout << std::left << std::setw(10) << l
                  << std::right << std::setw(20) << out_hetero_edges_per_layer[l]
                  << std::right << std::setw(20) << out_homo_edges_per_layer[l]
                  << std::right << std::setw(20) << out_hetero_edges_per_layer[l] + out_homo_edges_per_layer[l]
                  << std::right << std::setw(14) << std::fixed << std::setprecision(2) << (out_hetero_ratio_per_layer[l] * 100.0) << "%"
                  << std::right << std::setw(18) << std::fixed << std::setprecision(2) << (out_effective_edge_ratio_per_layer[l] * 100.0) << "%"
                  << std::endl;
    }
    
    std::cout << "========================================" << std::endl;
}

// ===================== 查询点生成 =====================
void generate_queries_from_categories(int query_count, unsigned int query_seed) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Step 5: Generating Queries From Category Groups" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<hnswlib::labeltype> all_labels;
    for (const auto& pair : category_to_labels) {
        all_labels.insert(all_labels.end(), pair.second.begin(), pair.second.end());
    }
    if (all_labels.empty()) throw std::runtime_error("No category data found for query generation");

    std::mt19937 gen(query_seed);
    std::shuffle(all_labels.begin(), all_labels.end(), gen);

    query_indices.clear();
    for (int i = 0; i < query_count && i < static_cast<int>(all_labels.size()); ++i) {
        query_indices.push_back(static_cast<int>(all_labels[i]));
    }
    std::cout << " Successfully generated " << query_indices.size()
              << " fixed query points (from labeled file category groups)" << std::endl;
}

// ===================== Brute-force：按偏差类别过滤 =====================
// ===================== Precompute category geometry for rho(Q) =====================
void precompute_category_geometry(
    int dim,
    const std::vector<float>& data_buffer
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Precomputing Category Geometry for Query Correlation" << std::endl;
    std::cout << "========================================" << std::endl;

    category_geometry_stats.clear();
    category_geometry_stats.resize(static_cast<size_t>(global_num_clusters));

    for (auto& stats : category_geometry_stats) {
        stats.mean.assign(static_cast<size_t>(dim), 0.0);
        stats.count = 0;
        stats.m2 = 0.0;
    }

    // Multivariate Welford update avoids numerical cancellation while
    // computing the sum of squared distances to the category mean.
    for (uint64_t i = 0; i < global_num_points; ++i) {
        const int category = point_categories[static_cast<size_t>(i)];
        if (category < 0 || category >= static_cast<int>(global_num_clusters)) {
            throw std::runtime_error("Invalid category while computing category geometry.");
        }

        CategoryGeometryStats& stats =
            category_geometry_stats[static_cast<size_t>(category)];
        const float* vector =
            data_buffer.data() + static_cast<size_t>(i) * static_cast<size_t>(dim);

        ++stats.count;
        const double inverse_count = 1.0 / static_cast<double>(stats.count);
        double m2_increment = 0.0;

        for (int d = 0; d < dim; ++d) {
            const double value = static_cast<double>(vector[d]);
            const double delta = value - stats.mean[static_cast<size_t>(d)];
            stats.mean[static_cast<size_t>(d)] += delta * inverse_count;
            const double delta_after_update = value - stats.mean[static_cast<size_t>(d)];
            m2_increment += delta * delta_after_update;
        }

        stats.m2 += m2_increment;
    }

    std::cout << " Category geometry computed for "
              << category_geometry_stats.size() << " categories" << std::endl;
}

// Select the qualified category by the query-to-centroid distance rank.
// bias_degree == 0 keeps the query's original category; bias_degree == 1
// selects the category whose centroid is farthest from the query vector.
int get_distance_biased_category(
    const float* query_vector,
    int original_category,
    float bias_degree,
    int dim
) {
    if (global_num_clusters == 0) {
        throw std::runtime_error("global_num_clusters is 0, cannot compute biased category.");
    }
    if (original_category < 0 ||
        original_category >= static_cast<int>(global_num_clusters)) {
        throw std::runtime_error("Invalid original category for distance-biased computation.");
    }
    if (bias_degree < 0.0f || bias_degree > 1.0f) {
        throw std::runtime_error("BIAS_DEGREE must be in [0, 1]!");
    }
    if (category_geometry_stats.size() != static_cast<size_t>(global_num_clusters)) {
        throw std::runtime_error("Category geometry must be computed before biased category selection.");
    }
    if (bias_degree == 0.0f || global_num_clusters == 1) {
        return original_category;
    }

    struct CategoryDistance {
        double distance2;
        int category;
    };

    std::vector<CategoryDistance> candidates;
    candidates.reserve(static_cast<size_t>(global_num_clusters - 1));

    for (int category = 0;
         category < static_cast<int>(global_num_clusters);
         ++category) {
        if (category == original_category) {
            continue;
        }

        const CategoryGeometryStats& stats =
            category_geometry_stats[static_cast<size_t>(category)];
        if (stats.count == 0 || stats.mean.size() != static_cast<size_t>(dim)) {
            continue;
        }

        double distance2 = 0.0;
        for (int d = 0; d < dim; ++d) {
            const double difference =
                static_cast<double>(query_vector[d])
                - stats.mean[static_cast<size_t>(d)];
            distance2 += difference * difference;
        }

        candidates.push_back({distance2, category});
    }

    if (candidates.empty()) {
        return original_category;
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const CategoryDistance& left, const CategoryDistance& right) {
            if (left.distance2 != right.distance2) {
                return left.distance2 < right.distance2;
            }
            return left.category < right.category;
        }
    );

    // Rank 0 denotes the original category. Ranks 1..C-1 denote the other
    // categories ordered from nearest to farthest by their centroid distance.
    const size_t rank = static_cast<size_t>(std::llround(
        static_cast<double>(bias_degree)
        * static_cast<double>(candidates.size())
    ));

    if (rank == 0) {
        return original_category;
    }

    return candidates[std::min(rank, candidates.size()) - 1].category;
}

// Compute the paper's filtered query correlation rho(Q) for every query.
// The qualified set X_Q is the complete vector set of the biased category.
double prepare_query_correlations(
    int dim,
    float bias_degree,
    const std::vector<float>& data_buffer,
    size_t& valid_query_count,
    size_t& negative_query_count
) {
    if (category_geometry_stats.size() != static_cast<size_t>(global_num_clusters)) {
        throw std::runtime_error("Category geometry must be computed before query correlations.");
    }

    query_correlation_records.clear();
    query_correlation_records.reserve(query_indices.size());

    valid_query_count = 0;
    negative_query_count = 0;
    double rho_sum = 0.0;

    for (size_t idx = 0; idx < query_indices.size(); ++idx) {
        const int query_index = query_indices[idx];
        const int original_category =
            point_categories[static_cast<size_t>(query_index)];
        const float* query_vector =
            data_buffer.data()
            + static_cast<size_t>(query_index) * static_cast<size_t>(dim);
        const int query_category =
            get_distance_biased_category(
                query_vector,
                original_category,
                bias_degree,
                dim
            );

        const CategoryGeometryStats& stats =
            category_geometry_stats[static_cast<size_t>(query_category)];

        QueryCorrelationRecord record;
        record.query_index = query_index;
        record.original_category = original_category;
        record.query_category = query_category;
        record.qualified_count = stats.count;

        if (stats.count > 0) {
            double centroid_distance2 = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double difference =
                    static_cast<double>(query_vector[d])
                    - stats.mean[static_cast<size_t>(d)];
                centroid_distance2 += difference * difference;
            }

            double sigma2 = stats.m2 / static_cast<double>(stats.count);
            if (sigma2 < 0.0 && sigma2 > -1e-12) {
                sigma2 = 0.0;
            }

            record.sigma2 = sigma2;
            record.query_centroid_distance2 = centroid_distance2;

            const double denominator = sigma2 + centroid_distance2;
            if (sigma2 >= 0.0 && denominator > 0.0 && std::isfinite(denominator)) {
                double rho = (sigma2 - centroid_distance2) / denominator;
                rho = std::max(-1.0, std::min(1.0, rho));

                record.rho = rho;
                record.valid = true;
                // std::cout << "rho: " << rho << "\n";
                rho_sum += rho;
                ++valid_query_count;
                if (rho < 0.0) {
                    ++negative_query_count;
                }
            }
        }

        query_correlation_records.push_back(record);
    }

    return valid_query_count > 0
        ? rho_sum / static_cast<double>(valid_query_count)
        : std::numeric_limits<double>::quiet_NaN();
}

std::vector<hnswlib::labeltype> category_brute_knn(
    const float* query_vec,
    int query_cat,
    int k,
    int dim,
    const std::vector<float>& data_buffer
) {
    auto it = category_to_labels.find(query_cat);
    if (it == category_to_labels.end()) {
        return {};
    }

    const auto& candidate_labels = it->second;
    std::vector<std::pair<float, hnswlib::labeltype>> dist_pairs;
    dist_pairs.reserve(candidate_labels.size());

    for (hnswlib::labeltype label : candidate_labels) {
        const float* vec = data_buffer.data() + static_cast<size_t>(label) * dim;
        float dist = global_dist_func(query_vec, vec, global_dist_func_param);
        dist_pairs.emplace_back(dist, label);
        total_brute_dist_calc++;
    }

    std::sort(dist_pairs.begin(), dist_pairs.end(),
              [](const std::pair<float, hnswlib::labeltype>& a,
                 const std::pair<float, hnswlib::labeltype>& b) {
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
              });

    std::vector<hnswlib::labeltype> results;
    int topk = std::min(k, static_cast<int>(dist_pairs.size()));
    results.reserve(topk);
    for (int i = 0; i < topk; ++i) {
        results.push_back(dist_pairs[i].second);
    }
    return results;
}

// ===================== 预计算 brute-force结果 =====================
void precompute_brute_force_results(
    int dim,
    int k,
    float BIAS_DEGREE,
    const std::vector<float>& data_buffer
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Step 6: Precomputing Biased Category Bruteforce Results" << std::endl;
    std::cout << "========================================" << std::endl;

    brute_force_cache.clear();
    total_brute_dist_calc = 0;
    const int query_total = static_cast<int>(query_indices.size());
    double total_brute_time = 0.0;

    if (query_correlation_records.size() != query_indices.size()) {
        throw std::runtime_error("Query correlations must be prepared before brute-force search.");
    }

    std::cout << "Running biased category brute-force search (total queries: "
              << query_total << ", k=" << k
              << ", BIAS_DEGREE=" << BIAS_DEGREE << ")..." << std::endl;

    for (int idx = 0; idx < query_total; ++idx) {
        const QueryCorrelationRecord& correlation_record =
            query_correlation_records[static_cast<size_t>(idx)];
        int query_idx = correlation_record.query_index;
        int original_cat = correlation_record.original_category;
        int search_cat = correlation_record.query_category;

        if (idx % 10 == 0) {
            std::cout << "\rProgress: Query " << idx + 1 << " / " << query_total
                      << " (OriginalCat: " << original_cat
                      << ", BiasedSearchCat: " << search_cat << ")" << std::flush;
        }

        const float* query_vec = data_buffer.data() + static_cast<size_t>(query_idx) * dim;
        auto start = std::chrono::steady_clock::now();
        auto results = category_brute_knn(query_vec, search_cat, k, dim, data_buffer);
        auto end = std::chrono::steady_clock::now();

        double time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        total_brute_time += time_ms;
        brute_force_cache[static_cast<hnswlib::labeltype>(query_idx)] = results;
    }

    std::cout << "\rProgress: Query " << query_total << " / " << query_total << " (Search Completed)" << std::flush;

    brute_force_avg_time = (query_total > 0) ? (total_brute_time / query_total) : 0.0;
    std::cout << "\n Bruteforce search completed!" << std::endl;
}

// ===================== 性能评估 (变量名更新为 Champion Layer 语义) =====================
void evaluate_search_performance(
    hnswlib::HierarchicalNSW<float>* alg_hnsw,
    uint64_t max_elements,
    int dim,
    int k,
    float BIAS_DEGREE,
    const std::vector<int>& efSearchs,
    int ef_construction,
    int M,
    const std::vector<float>& data_buffer,
    double& out_avg_brute_dist,
    std::vector<double>& out_recall,
    std::vector<double>& out_hnsw_time,
    std::vector<double>& out_search_dist,
    std::vector<double>& out_dist_opt,
    std::vector<double>& out_time_opt,
    std::vector<double>& out_champion_layer_rewrite_dist, // 【修改7】改名
    int hop_count,
    int init_hop_count,
    int query_rewrite_method,
    int rgn_rewrite_hop_count,
    std::vector<std::unordered_map<uint16_t, int>>& out_top_candidates_stats,
    bool& out_all_stats_same,
    std::vector<double>& out_overall_blank_rates,
    std::vector<double>& out_average_nfr_rates,
    std::vector<std::vector<std::pair<uint16_t, double>>>& out_blank_rates_details_list
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Step 7: Evaluating HNSW Search Performance (Biased Category)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Evaluation Config:" << std::endl;
    std::cout << "  - M: " << M << std::endl;
    std::cout << "  - ef_construction: " << ef_construction << std::endl;
    std::cout << "  - ef_search: ";
    for (size_t i = 0; i < efSearchs.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << efSearchs[i];
    }
    std::cout << std::endl;
    std::cout << "  - k (Recall@k): " << k << std::endl;
    std::cout << "  - BIAS_DEGREE: " << BIAS_DEGREE << " (0=无偏差,1=最大偏差)" << std::endl;
    std::cout << "  - Category Source: WIKI_label.bin" << std::endl;
    std::cout << "  - Dataset: " << max_elements << " elements, " << dim << " dimensions" << std::endl;

    const int query_total = static_cast<int>(query_indices.size());
    if (query_correlation_records.size() != query_indices.size()) {
        throw std::runtime_error("Query correlations must be prepared before HNSW evaluation.");
    }
    const double avg_brute_dist = (query_total > 0)
        ? static_cast<double>(total_brute_dist_calc) / query_total
        : 0.0;
    out_avg_brute_dist = avg_brute_dist;

    std::vector<double> vec_recall;
    std::vector<double> vec_hnsw_time;
    std::vector<double> vec_search_dist;
    std::vector<double> vec_dist_opt;
    std::vector<double> vec_time_opt;
    std::vector<double> vec_champion_layer_rewrite_dist; // 【修改8】改名

    out_top_candidates_stats.resize(efSearchs.size());
    out_overall_blank_rates.resize(efSearchs.size());
    out_average_nfr_rates.resize(efSearchs.size());
    out_blank_rates_details_list.resize(efSearchs.size());

    std::cout << "\nRunning biased category HNSW search..." << std::endl;

    for (size_t ef_idx = 0; ef_idx < efSearchs.size(); ++ef_idx) {
        int ef = efSearchs[ef_idx];
        alg_hnsw->setEf(ef);

        double total_recall = 0.0;
        double total_hnsw_time = 0.0;
        long long total_search_dist_calc = 0;
        uint64_t total_champion_layer_rewrite_dist_calc = 0; // 【修改9】改名

        auto& current_stats_map = out_top_candidates_stats[ef_idx];

        for (int idx = 0; idx < query_total; ++idx) {
            const QueryCorrelationRecord& correlation_record =
                query_correlation_records[static_cast<size_t>(idx)];
            int query_index = correlation_record.query_index;
            int original_category = correlation_record.original_category;
            int query_category = correlation_record.query_category;

            auto label = static_cast<hnswlib::labeltype>(query_index);
            const auto& brute_labels = brute_force_cache.at(label);
            const float* query_vec = data_buffer.data() + static_cast<size_t>(query_index) * dim;

            alg_hnsw->resetSearchDistanceComputations();
            alg_hnsw->resetChampionRewriteDistanceComputations(); // 逻辑保留，库函数内部实现已变
            alg_hnsw->resetSearchTopCandidatesCount();

            auto hnsw_start = std::chrono::steady_clock::now();
            auto hnsw_results = alg_hnsw->searchKnn(query_vec, k, query_category, hop_count, init_hop_count, query_rewrite_method, rgn_rewrite_hop_count);
            auto hnsw_end = std::chrono::steady_clock::now();

            uint16_t current_count = alg_hnsw->getSearchTopCandidatesCount();
            current_stats_map[current_count]++;

            double hnsw_time = std::chrono::duration_cast<std::chrono::microseconds>(hnsw_end - hnsw_start).count() / 1000.0;
            total_hnsw_time += hnsw_time;
            total_search_dist_calc += alg_hnsw->getSearchDistanceComputations();
            total_champion_layer_rewrite_dist_calc += alg_hnsw->getChampionRewriteDistanceComputations(); // 【修改10】改名

            std::vector<hnswlib::labeltype> hnsw_labels;
            hnsw_labels.reserve(hnsw_results.size());
            for (const auto& p : hnsw_results) {
                hnsw_labels.push_back(p.first);
            }

            int denom = std::max(1, std::min(k, static_cast<int>(brute_labels.size())));
            int overlap = 0;
            std::unordered_set<hnswlib::labeltype> brute_set(brute_labels.begin(), brute_labels.end());
            for (const auto& lbl : hnsw_labels) {
                if (brute_set.count(lbl)) overlap++;
            }
            total_recall += static_cast<double>(overlap) / denom;
        }

        std::cout << "\refSearch=" << ef
                  << ":Progress: Query " << query_total
                  << " / " << query_total
                  << " (Search Completed)    " << std::endl;

        double avg_recall = (query_total > 0) ? (total_recall / query_total) : 0.0;
        double avg_hnsw_time = (query_total > 0) ? (total_hnsw_time / query_total) : 0.0;
        double avg_search_dist = (query_total > 0) ? (static_cast<double>(total_search_dist_calc) / query_total) : 0.0;
        double avg_champion_layer_rewrite_dist = (query_total > 0) ? (static_cast<double>(total_champion_layer_rewrite_dist_calc) / query_total) : 0.0; // 【修改11】改名

        double dist_opt = (avg_search_dist > 0.0) ? (((avg_brute_dist / avg_search_dist) - 1) * 100.0) : 0.0;
        double time_opt = (avg_hnsw_time > 0.0) ? (((brute_force_avg_time / avg_hnsw_time) - 1) * 100.0) : 0.0;

        vec_recall.push_back(avg_recall);
        vec_hnsw_time.push_back(avg_hnsw_time);
        vec_search_dist.push_back(avg_search_dist);
        vec_dist_opt.push_back(dist_opt);
        vec_time_opt.push_back(time_opt);
        vec_champion_layer_rewrite_dist.push_back(avg_champion_layer_rewrite_dist); // 【修改12】改名
    }

    out_recall = vec_recall;
    out_hnsw_time = vec_hnsw_time;
    out_search_dist = vec_search_dist;
    out_dist_opt = vec_dist_opt;
    out_time_opt = vec_time_opt;
    out_champion_layer_rewrite_dist = vec_champion_layer_rewrite_dist; // 【修改13】改名

    out_all_stats_same = true;
    
    if (efSearchs.size() > 1) {
        const auto& base_stats = out_top_candidates_stats[0];
        const size_t base_pool_capacity = std::max(
            static_cast<size_t>(efSearchs[0]),
            static_cast<size_t>(k)
        );

        std::unordered_map<uint16_t, int> base_non_max;
        for (const auto& pair : base_stats) {
            if (static_cast<size_t>(pair.first) < base_pool_capacity) {
                base_non_max[pair.first] = pair.second;
            }
        }

        for (size_t i = 1; i < out_top_candidates_stats.size(); ++i) {
            const auto& current_stats = out_top_candidates_stats[i];
            const size_t current_pool_capacity = std::max(
                static_cast<size_t>(efSearchs[i]),
                static_cast<size_t>(k)
            );

            std::unordered_map<uint16_t, int> curr_non_max;
            for (const auto& pair : current_stats) {
                if (static_cast<size_t>(pair.first) < current_pool_capacity) {
                    curr_non_max[pair.first] = pair.second;
                }
            }

            if (curr_non_max != base_non_max) {
                out_all_stats_same = false;
                std::cerr << "\nERROR: Stats mismatch detected!" << std::endl;
            }
        }
    }

    for (size_t i = 0; i < out_top_candidates_stats.size(); ++i) {
        const auto& stats = out_top_candidates_stats[i];
        const size_t pool_capacity = std::max(
            static_cast<size_t>(efSearchs[i]),
            static_cast<size_t>(k)
        );

        int full_query_count = 0;
        double weighted_nfr_sum = 0.0;

        out_blank_rates_details_list[i].clear();

        for (const auto& pair : stats) {
            const size_t working_pool_size = std::min(
                static_cast<size_t>(pair.first),
                pool_capacity
            );
            const int query_frequency = pair.second;

            // Paper definition: NFR(Q) = 1 - |W_Q| / B.
            const double query_nfr =
                1.0
                - static_cast<double>(working_pool_size)
                    / static_cast<double>(pool_capacity);

            weighted_nfr_sum +=
                static_cast<double>(query_frequency) * query_nfr;

            if (working_pool_size >= pool_capacity) {
                full_query_count += query_frequency;
            } else {
                const double rate =
                    query_total > 0
                        ? static_cast<double>(query_frequency) * 100.0
                            / static_cast<double>(query_total)
                        : 0.0;
                out_blank_rates_details_list[i].emplace_back(pair.first, rate);
            }
        }

        // Percentage of queries whose qualified working pool is not full.
        out_overall_blank_rates[i] =
            query_total > 0
                ? static_cast<double>(query_total - full_query_count) * 100.0
                    / static_cast<double>(query_total)
                : 0.0;

        // Average of NFR(Q) over all queries, expressed as a percentage.
        out_average_nfr_rates[i] =
            query_total > 0
                ? weighted_nfr_sum * 100.0
                    / static_cast<double>(query_total)
                : 0.0;

        std::sort(
            out_blank_rates_details_list[i].begin(),
            out_blank_rates_details_list[i].end(),
            [](const std::pair<uint16_t, double>& left,
               const std::pair<uint16_t, double>& right) {
                return left.first < right.first;
            }
        );
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Step 8: Evaluation Results (Biased Category HNSW)" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "1. HNSW Index Construction Metrics" << std::endl;
    std::cout << "HNSW Index Build Time: " << hnsw_build_time << "ms" << std::endl;
    std::cout << "Total Distance Calculations (Build): " << hnsw_build_dist_calc << std::endl;
    std::cout << "Champion Layer Distance Calculations (Build): "
              << champion_layer_build_dist_calc << std::endl;

    std::cout << "\n2. Baseline Metrics (Brute-force)" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Avg Time per Query: " << brute_force_avg_time << "ms" << std::endl;
    std::cout << "Avg Distance Calculations per Query: " << avg_brute_dist << std::endl;

    std::cout << "\n3. Query Performance Metrics (efSearch Table)" << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(10) << "efSearch"
              << std::right << std::setw(15) << "Blank Rate"
              << std::right << std::setw(15) << "Avg NFR"
              << std::right << std::setw(20) << "Average Recall@10"
              << std::right << std::setw(20) << "HNSW Avg Time    "
              << std::right << std::setw(25) << "HNSW Avg Dist Calculations    "
              << std::right << std::setw(30) << "Champion Layer Rewrite Avg Dist Calc    " // 【修改15】改名
              << std::right << std::setw(25) << "Dist Calculation Optimization    "
              << std::right << std::setw(25) << "Query Time Optimization"
              << std::endl;
    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < efSearchs.size(); ++i) {
        std::cout << std::left << std::setw(10) << ("ef=" + std::to_string(efSearchs[i]))
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << out_overall_blank_rates[i] << "%"
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) << out_average_nfr_rates[i] << "%"
                  << std::right << std::setw(18) << std::fixed << std::setprecision(2) << vec_recall[i] * 100 << "%"
                  << std::right << std::setw(17) << std::fixed << std::setprecision(3) << vec_hnsw_time[i] << "ms"
                  << std::right << std::setw(24) << std::fixed << std::setprecision(1) << vec_search_dist[i]
                  << std::right << std::setw(29) << std::fixed << std::setprecision(1) << vec_champion_layer_rewrite_dist[i] // 【修改16】改名
                  << std::right << std::setw(28) << std::fixed << std::setprecision(2) << vec_dist_opt[i] << "%"
                  << std::right << std::setw(33) << std::fixed << std::setprecision(2) << vec_time_opt[i] << "%"
                  << std::endl;
    }

    std::cout << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ===================== CSV保存函数 (更新为 Champion Layer 参数) =====================
void save_results_to_csv(
    const std::string& csv_file_path,
    const std::vector<int>& efSearchs,
    int M,
    int ef_construction,
    int k,
    int dim,
    uint64_t max_elements,
    const std::string& data_path,
    int query_count,
    unsigned int query_seed,
    float BIAS_DEGREE,
    double average_query_rho,
    size_t valid_rho_query_count,
    size_t negative_rho_query_count,
    double negative_query_ratio_percent,
    uint32_t num_clusters,
    int hop_count,
    int init_hop_count,
    int query_rewrite_method,
    int rgn_rewrite_hop_count,
    int acorn_build_mode,
    size_t acorn_gamma,
    size_t acorn_m_beta,
    int champion_layer_level,
    double champion_layer_ratio,
    size_t champion_layer_min_per_category,
    size_t champion_rewrite_topk,
    double brute_avg_time_ms,
    double brute_avg_dist_calc,
    const std::vector<double>& recall,
    const std::vector<double>& hnsw_avg_time_ms,
    const std::vector<double>& hnsw_avg_dist_calc,
    const std::vector<double>& champion_layer_rewrite_avg_dist_calc,
    const std::vector<double>& dist_optimization_percent,
    const std::vector<double>& time_optimization_percent,
    const std::vector<double>& overall_blank_rates,
    const std::vector<double>& average_nfr_rates,
    const std::vector<std::vector<std::pair<uint16_t, double>>>& blank_rates_details_list,
    const std::vector<double>& layer_hetero_ratios,
    const std::vector<double>& layer_effective_edge_ratios,
    int heuristic_control
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Step 9: Saving Performance Results to CSV" << std::endl;
    std::cout << "========================================" << std::endl;

    std::ofstream csv_file(csv_file_path, std::ios::out | std::ios::trunc);
    if (!csv_file.is_open()) {
        std::cerr << "ERROR: Failed to create CSV file!" << std::endl;
        return;
    }

    // 【修改21】CSV Header 更新：去掉 champion_list_capacity 和 champion_build_distance_calc，增加 layer 参数
    csv_file << "efSearch,M,efConstruction,k,dimension,max_elements,data_file,"
                "query_count,query_seed,BIAS_DEGREE,avg_filtered_query_correlation_rho,"
                "valid_rho_query_count,negative_rho_query_count,negative_query_ratio_percent,"
                "num_clusters,hop_count,init_hop_count,"
                "query_rewrite_method,rgn_rewrite_hop_count,"
                "acorn_build_mode,acorn_gamma,acorn_m_beta,"
                "champion_layer_level,champion_layer_ratio,champion_layer_min_per_category,champion_rewrite_topk,"
                "recall_percent,hnsw_avg_query_time_ms,hnsw_avg_distance_calc,"
                "champion_layer_rewrite_avg_distance_calc,"
                "brute_avg_query_time_ms,brute_avg_distance_calc,"
                "distance_optimization_percent,time_optimization_percent,"
                "hnsw_build_time_ms,hnsw_build_total_distance_calc,champion_layer_build_distance_calc,"
                "overall_blank_rate_percent,average_nfr_percent,blank_rates_details,"
                "layer_hetero_ratios,layer_effective_edge_ratios,heuristic_control\n";

    std::string layer_ratios_str;
    for (size_t l = 0; l < layer_hetero_ratios.size(); ++l) {
        if (l > 0) layer_ratios_str += ";";
        std::ostringstream oss;
        oss << l << ":" << std::fixed << std::setprecision(4) << layer_hetero_ratios[l];
        layer_ratios_str += oss.str();
    }

    std::string layer_effective_ratios_str;
    for (size_t l = 0; l < layer_effective_edge_ratios.size(); ++l) {
        if (l > 0) layer_effective_ratios_str += ";";
        std::ostringstream oss;
        oss << l << ":" << std::fixed << std::setprecision(4) << layer_effective_edge_ratios[l];
        layer_effective_ratios_str += oss.str();
    }

    csv_file << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < efSearchs.size(); ++i) {
        csv_file << efSearchs[i] << ","
                 << M << ","
                 << ef_construction << ","
                 << k << ","
                 << dim << ","
                 << max_elements << ","
                 << data_path << ","
                 << query_count << ","
                 << query_seed << ","
                 << BIAS_DEGREE << ","
                 << std::setprecision(8) << average_query_rho << ","
                 << valid_rho_query_count << ","
                 << negative_rho_query_count << ","
                 << std::setprecision(4) << negative_query_ratio_percent << ","
                 << num_clusters << ","
                 << hop_count << ","
                 << init_hop_count << ","
                 << query_rewrite_method << ","
                 << rgn_rewrite_hop_count << ","
                 << acorn_build_mode << ","
                 << acorn_gamma << ","
                 << acorn_m_beta << ","
                 << champion_layer_level << "," // 【修改22】写入新参数
                 << champion_layer_ratio << ","
                 << champion_layer_min_per_category << "," // 【修改23】写入新参数
                 << champion_rewrite_topk << ","
                 << recall[i] * 100 << ","
                 << hnsw_avg_time_ms[i] << ","
                 << hnsw_avg_dist_calc[i] << ","
                 << champion_layer_rewrite_avg_dist_calc[i] << "," // 【修改24】改名
                 << brute_avg_time_ms << ","
                 << brute_avg_dist_calc << ","
                 << dist_optimization_percent[i] << ","
                 << time_optimization_percent[i] << ","
                 << hnsw_build_time << ","
                 << hnsw_build_dist_calc << ","
                 << champion_layer_build_dist_calc << ",";

        csv_file << overall_blank_rates[i] << ",";
        csv_file << average_nfr_rates[i] << ",";
        
        std::string details_str;
        for (size_t j = 0; j < blank_rates_details_list[i].size(); ++j) {
            if (j > 0) details_str += ";";
            std::ostringstream oss;
            oss << blank_rates_details_list[i][j].first << ":" << std::fixed << std::setprecision(2) << blank_rates_details_list[i][j].second << "%";
            details_str += oss.str();
        }
        csv_file << details_str << ",";
        
        csv_file << layer_ratios_str << ",";
        csv_file << layer_effective_ratios_str << ",";
        csv_file << heuristic_control;

        csv_file << "\n";
    }

    csv_file.close();
    std::cout << "CSV file saved successfully: " << csv_file_path << std::endl;
    std::cout << "========================================" << std::endl;
}

// ===================== 统一实验参数（默认值 + 命令行覆盖） =====================
// 无命令行参数时，程序完全使用此处默认值。
// 如需长期更换默认实验，只需修改这一个结构体；
// 如需临时做批量实验，可用同名命令行选项覆盖它们。
struct ProgramOptions {
    // ---------- 输入与输出 ----------
    // 带文件头、向量和聚类标签的输入数据集。
    std::string data_path =
        "D:\\secondpaper\\Data\\INPUT\\WIKI\\WIKI_label10_100_10e6.bin";
    // HNSW/ACORN 主索引文件；构建参数不同时必须使用不同文件名。
    std::string hnsw_index_path =
        "D:\\secondpaper\\Data\\OUTPUT\\BIN\\paperexperients\\pai\\acorn-pMefc200c10C100WIKIM16m10e6.bin";
    // 与主索引配套的构建时间、距离计算量等元数据（不是暴力搜索引）。
    std::string brute_index_path =
        "D:\\secondpaper\\Data\\OUTPUT\\BIN\\paperexperients\\pai\\brute-pMefc200c10C100WIKIM16m10e6.bin";
    // 本次查询实验的 CSV 结果文件。
    std::string csv_save_path =
        "D:\\secondpaper\\Data\\OUTPUT\\CSV\\paperexperients\\pai\\rgnacorn-pMefc200c10Rh1B0.75i1h2WIKIM16m10e6.csv";

    // ---------- HNSW/ACORN 索引构建参数 ----------
    int M = 16;                         // HNSW 每个节点的基础连接度参数。
    int ef_construction = 200;          // 建图候选集宽度；越大通常索引质量越高，构建越慢。
    size_t index_random_seed = 100;     // HNSW 随机分层的种子，用于保证构建可复现。
    bool allow_replace_deleted = false; // 是否允许新节点复用被删除节点的位置；本实验不删点。
    int acorn_build_mode = 0;           // 0=ACORN-1；1=未压缩 ACORN-gamma。
    size_t acorn_gamma = 2;             // ACORN-gamma 度数放大倍数；mode=1 时生效，且必须 >=2。
    int heuristic_control = 0;          // 0/1，控制底层邻居选择启发式。

    // ---------- 冠军层（Champion Layer）构建参数 ----------
    int champion_layer_level = 1;                // 冠军层起始层 lc，必须 >=1。
    double champion_layer_ratio = -1.0;          // -1=原生 HNSW 随机分层；[0,1]=level>=lc 的目标比例。
    size_t champion_layer_min_per_category = 100;// 每类在冠军层中至少保留的节点数 R；0=不额外保底。

    // ---------- 查询与评估参数 ----------
    int k = 10;                                  // 返回近邻数，并计算 Recall@k。
    std::vector<int> ef_search_values =          // 需要依次测试的 efSearch 列表，每个值产生一行 CSV。
        {10, 20, 50, 100, 200, 300, 500, 800, 1000, 2000};
    int query_count = 100;                       // 从数据集中抽取的查询数量。
    unsigned int query_seed = 4;                 // 查询抽样随机种子，相同种子保证各方法使用同一批查询。
    float bias_degree = 0.75f;                   // 过滤类别偏差程度：0=无偏差，1=最大偏差。
    int hop_count = 2;                           // 底层过滤搜索的扩展跳数。
    int init_hop_count = 1;                      // 底层初始候选点的扩展跳数，可为 0。
    bool enable_query_rewrite = true;            // true=开启查询改写（RGN/ORGN/均值）；false=关闭。
    int query_rewrite_method = 2;                // 0=候选点均值；1=ORGN 全局加权；2=RGN 局部多跳加权。
    int rgn_rewrite_hop_count = 1;               // method=2 时，RGN 在冠军层的局部扩展跳数。
    size_t champion_rewrite_topk = 10;           // method=0 时用于查询改写的冠军候选点数。

    // ---------- 文件与运行控制 ----------
    bool save_index_file = true;        // 新建索引后是否保存主索引和构建元数据。
    bool save_csv = true;               // 是否保存查询评估 CSV。
    bool show_help = false;             // 仅打印命令行帮助。
    bool dry_run = false;               // 仅检查并打印参数，不读取数据、不实验。
    bool skip_if_csv_complete = false;  // CSV 已含完整 efSearch 结果时直接跳过，用于断点续跑。
};

std::string normalize_option_name(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
        if (c == '_') {
            return '-';
        }
        return static_cast<char>(std::tolower(c));
    });
    return name;
}

int parse_int_option(const std::string& option, const std::string& value) {
    size_t parsed_chars = 0;
    int parsed_value = 0;
    try {
        parsed_value = std::stoi(value, &parsed_chars);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " expects an integer, got: " + value);
    }
    if (parsed_chars != value.size()) {
        throw std::invalid_argument(option + " expects an integer, got: " + value);
    }
    return parsed_value;
}

size_t parse_size_option(const std::string& option, const std::string& value) {
    if (!value.empty() && value.front() == '-') {
        throw std::invalid_argument(option + " expects a non-negative integer, got: " + value);
    }
    size_t parsed_chars = 0;
    unsigned long long parsed_value = 0;
    try {
        parsed_value = std::stoull(value, &parsed_chars);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " expects a non-negative integer, got: " + value);
    }
    if (parsed_chars != value.size() ||
        parsed_value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
        throw std::invalid_argument(option + " expects a valid non-negative integer, got: " + value);
    }
    return static_cast<size_t>(parsed_value);
}

double parse_double_option(const std::string& option, const std::string& value) {
    size_t parsed_chars = 0;
    double parsed_value = 0.0;
    try {
        parsed_value = std::stod(value, &parsed_chars);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " expects a number, got: " + value);
    }
    if (parsed_chars != value.size() || !std::isfinite(parsed_value)) {
        throw std::invalid_argument(option + " expects a finite number, got: " + value);
    }
    return parsed_value;
}

std::vector<int> parse_int_list_option(
    const std::string& option,
    const std::string& value
) {
    std::vector<int> parsed_values;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::invalid_argument(
                option + " expects comma-separated integers, got: " + value);
        }
        parsed_values.push_back(parse_int_option(option, item));
    }
    if (parsed_values.empty()) {
        throw std::invalid_argument(
            option + " expects at least one integer, got: " + value);
    }
    return parsed_values;
}

bool parse_bool_option(const std::string& option, std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    throw std::invalid_argument(
        option + " expects one of 0/1, true/false, yes/no, or on/off, got: " + value);
}

int parse_acorn_build_mode(const std::string& option, std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::replace(value.begin(), value.end(), '_', '-');
    if (value == "0" || value == "acorn-1" || value == "acorn1") {
        return 0;
    }
    if (value == "1" || value == "acorn-gamma" || value == "acorn-g") {
        return 1;
    }
    throw std::invalid_argument(
        option + " expects 0/ACORN-1 or 1/ACORN-gamma, got: " + value);
}

void print_usage(const char* executable) {
    std::cout
        << "Usage:\n  " << executable << " [options]\n\n"
        << "Options (hyphens and underscores are interchangeable; names are case-insensitive):\n"
        << "  --data-path <path>                 Labeled input dataset\n"
        << "  --hnsw-index-path <path>           HNSW/ACORN index file\n"
        << "  --brute-index-path <path>          Build-metrics sidecar file\n"
        << "  --csv-save-path <path>             Experiment CSV output\n"
        << "  --m <integer>                      HNSW degree parameter M\n"
        << "  --ef-construction <integer>        Index construction beam width\n"
        << "  --index-random-seed <integer>      Random seed used to build index levels\n"
        << "  --allow-replace-deleted <boolean>  Allow reuse of deleted element slots\n"
        << "  --acorn-build-mode <mode>          0/ACORN-1 or 1/ACORN-gamma\n"
        << "  --acorn-gamma <integer>            Gamma for ACORN-gamma (default: 2)\n"
        << "  --heuristic-control <0|1>          Bottom-layer neighbor heuristic\n"
        << "  --champion-layer-level <integer>   Champion-layer starting level lc\n"
        << "  --champion-layer-ratio <number>    -1 or target ratio in [0,1]\n"
        << "  --champion-layer-min-per-category <integer>  Per-category minimum R\n"
        << "  --k <integer>                      Result count and Recall@k\n"
        << "  --ef-search-values <a,b,...>       Comma-separated efSearch values\n"
        << "  --query-count <integer>            Number of sampled queries\n"
        << "  --query-seed <integer>             Query sampling seed\n"
        << "  --bias-degree <number>             Filter-category bias in [0,1]\n"
        << "  --hop-count <integer>              Bottom-layer search expansion hops\n"
        << "  --init-hop-count <integer>         Initial bottom-layer seeding hops\n"
        << "  --enable-query-rewrite <boolean>   Enable/disable RGN query rewrite\n"
        << "  --query-rewrite-method <0|1|2>     0=mean, 1=ORGN, 2=RGN\n"
        << "  --rgn-rewrite-hop-count <integer>  RGN guidance-layer expansion hops\n"
        << "  --champion-rewrite-topk <integer>  Candidate count for rewrite method 0\n"
        << "  --save-index-file <boolean>        Save a newly built index and metrics\n"
        << "  --save-csv <boolean>               Save experiment results to CSV\n"
        << "  --skip-if-csv-complete             Skip when CSV has all result rows\n"
        << "  --dry-run                          Validate and print configuration only\n"
        << "  --help                             Show this help\n\n"
        << "All options are optional. Omitting them preserves the original defaults.\n";
}

ProgramOptions parse_program_options(int argc, char** argv) {
    ProgramOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string argument = argv[i];
        if (argument.rfind("--", 0) != 0) {
            throw std::invalid_argument("Unexpected positional argument: " + argument);
        }

        std::string option = argument;
        std::string value;
        const size_t equals_pos = argument.find('=');
        if (equals_pos != std::string::npos) {
            option = argument.substr(0, equals_pos);
            value = argument.substr(equals_pos + 1);
        }
        option = normalize_option_name(option);

        if (option == "--help") {
            if (equals_pos != std::string::npos) {
                throw std::invalid_argument("--help does not accept a value");
            }
            options.show_help = true;
            continue;
        }
        if (option == "--dry-run") {
            if (equals_pos == std::string::npos) {
                options.dry_run = true;
            } else {
                options.dry_run = parse_bool_option(option, value);
            }
            continue;
        }
        if (option == "--skip-if-csv-complete") {
            if (equals_pos == std::string::npos) {
                options.skip_if_csv_complete = true;
            } else {
                options.skip_if_csv_complete = parse_bool_option(option, value);
            }
            continue;
        }

        if (equals_pos == std::string::npos) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for option: " + option);
            }
            value = argv[++i];
        }

        if (option == "--data-path") {
            options.data_path = value;
        } else if (option == "--hnsw-index-path") {
            options.hnsw_index_path = value;
        } else if (option == "--brute-index-path") {
            options.brute_index_path = value;
        } else if (option == "--csv-save-path") {
            options.csv_save_path = value;
        } else if (option == "--m") {
            options.M = parse_int_option(option, value);
        } else if (option == "--ef-construction") {
            options.ef_construction = parse_int_option(option, value);
        } else if (option == "--index-random-seed") {
            options.index_random_seed = parse_size_option(option, value);
        } else if (option == "--allow-replace-deleted") {
            options.allow_replace_deleted = parse_bool_option(option, value);
        } else if (option == "--acorn-build-mode") {
            options.acorn_build_mode = parse_acorn_build_mode(option, value);
        } else if (option == "--acorn-gamma") {
            options.acorn_gamma = parse_size_option(option, value);
        } else if (option == "--heuristic-control") {
            options.heuristic_control = parse_int_option(option, value);
        } else if (option == "--champion-layer-level") {
            options.champion_layer_level = parse_int_option(option, value);
        } else if (option == "--champion-layer-ratio") {
            options.champion_layer_ratio = parse_double_option(option, value);
        } else if (option == "--champion-layer-min-per-category") {
            options.champion_layer_min_per_category = parse_size_option(option, value);
        } else if (option == "--k") {
            options.k = parse_int_option(option, value);
        } else if (option == "--ef-search-values" || option == "--ef-searchs") {
            options.ef_search_values = parse_int_list_option(option, value);
        } else if (option == "--query-count") {
            options.query_count = parse_int_option(option, value);
        } else if (option == "--query-seed") {
            const size_t parsed_seed = parse_size_option(option, value);
            if (parsed_seed > static_cast<size_t>(
                    std::numeric_limits<unsigned int>::max())) {
                throw std::invalid_argument(option + " is too large: " + value);
            }
            options.query_seed = static_cast<unsigned int>(parsed_seed);
        } else if (option == "--bias-degree") {
            options.bias_degree = static_cast<float>(
                parse_double_option(option, value));
        } else if (option == "--hop-count") {
            options.hop_count = parse_int_option(option, value);
        } else if (option == "--init-hop-count") {
            options.init_hop_count = parse_int_option(option, value);
        } else if (option == "--enable-query-rewrite") {
            options.enable_query_rewrite = parse_bool_option(option, value);
        } else if (option == "--query-rewrite-method") {
            options.query_rewrite_method = parse_int_option(option, value);
        } else if (option == "--rgn-rewrite-hop-count") {
            options.rgn_rewrite_hop_count = parse_int_option(option, value);
        } else if (option == "--champion-rewrite-topk") {
            options.champion_rewrite_topk = parse_size_option(option, value);
        } else if (option == "--save-index-file") {
            options.save_index_file = parse_bool_option(option, value);
        } else if (option == "--save-csv") {
            options.save_csv = parse_bool_option(option, value);
        } else {
            throw std::invalid_argument("Unknown option: " + option);
        }
    }

    if (options.data_path.empty() || options.hnsw_index_path.empty() ||
        options.brute_index_path.empty() || options.csv_save_path.empty()) {
        throw std::invalid_argument("Dataset, index, metrics, and CSV paths must not be empty");
    }
    if (options.M < 2) {
        throw std::invalid_argument("--m must be at least 2");
    }
    if (options.M > 10000) {
        throw std::invalid_argument(
            "--m must not exceed 10000 (the index implementation caps larger values)");
    }
    if (options.ef_construction < 1) {
        throw std::invalid_argument("--ef-construction must be at least 1");
    }
    if (options.heuristic_control != 0 && options.heuristic_control != 1) {
        throw std::invalid_argument("--heuristic-control must be 0 or 1");
    }
    if (options.champion_layer_level < 1) {
        throw std::invalid_argument("--champion-layer-level must be at least 1");
    }
    if (options.champion_layer_ratio != -1.0 &&
        (options.champion_layer_ratio < 0.0 ||
         options.champion_layer_ratio > 1.0)) {
        throw std::invalid_argument(
            "--champion-layer-ratio must be -1 or in [0, 1]");
    }
    if (options.k < 1) {
        throw std::invalid_argument("--k must be at least 1");
    }
    for (int ef_search : options.ef_search_values) {
        if (ef_search < options.k) {
            throw std::invalid_argument(
                "every --ef-search-values entry must be at least --k; "
                "otherwise the implementation silently uses max(efSearch, k)");
        }
    }
    if (options.query_count < 1) {
        throw std::invalid_argument("--query-count must be at least 1");
    }
    if (!std::isfinite(options.bias_degree) ||
        options.bias_degree < 0.0f || options.bias_degree > 1.0f) {
        throw std::invalid_argument("--bias-degree must be in [0, 1]");
    }
    if (options.hop_count < 1) {
        throw std::invalid_argument("--hop-count must be at least 1");
    }
    if (options.init_hop_count < 0) {
        throw std::invalid_argument("--init-hop-count must be non-negative");
    }
    if (options.rgn_rewrite_hop_count < 0) {
        throw std::invalid_argument("--rgn-rewrite-hop-count must be non-negative");
    }
    if (options.query_rewrite_method < 0 ||
        options.query_rewrite_method > 2) {
        throw std::invalid_argument("--query-rewrite-method must be 0, 1, or 2");
    }
    if (options.enable_query_rewrite &&
        options.query_rewrite_method == 0 &&
        options.champion_rewrite_topk < 1) {
        throw std::invalid_argument(
            "--champion-rewrite-topk must be at least 1 for rewrite method 0");
    }
    if (options.acorn_build_mode == 1 && options.acorn_gamma < 2) {
        throw std::invalid_argument("ACORN-gamma requires --acorn-gamma >= 2");
    }
    const size_t effective_gamma =
        options.acorn_build_mode == 1 ? options.acorn_gamma : 1;
    const size_t max_storable_neighbors =
        static_cast<size_t>(std::numeric_limits<unsigned short>::max());
    if (static_cast<size_t>(options.M) >
        max_storable_neighbors / (2 * effective_gamma)) {
        throw std::invalid_argument(
            "2 * M * gamma exceeds the index link-list capacity");
    }
    if (options.acorn_build_mode == 1 && options.hop_count != 1) {
        throw std::invalid_argument(
            "ACORN-gamma uses its expanded direct-neighbor list; set --hop-count 1");
    }

    return options;
}

bool csv_has_complete_results(
    const std::string& csv_path,
    size_t expected_result_rows
) {
    std::ifstream input(csv_path);
    if (!input) {
        return false;
    }

    std::string line;
    if (!std::getline(input, line) || line.rfind("efSearch,", 0) != 0) {
        return false;
    }

    size_t result_rows = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line != "\r") {
            ++result_rows;
        }
    }
    return result_rows >= expected_result_rows;
}

// ===================== 主函数 (配置参数与接口调用大改) =====================
int main(int argc, char** argv) {
    ProgramOptions options;
    try {
        options = parse_program_options(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (options.show_help) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    // 所有可调参数的默认值都在 main 上方的 ProgramOptions 中。
    // 这里只建立局部别名，不再定义任何独立默认值。
    const int M = options.M;
    const int k = options.k;
    const int ef_construction = options.ef_construction;
    const std::vector<int>& efSearchs = options.ef_search_values;
    const int query_count = options.query_count;
    const unsigned int query_seed = options.query_seed;

    const float BIAS_DEGREE = options.bias_degree;
    
    // 【修改26】配置参数大换血：冠军列表 -> 冠军层
    const bool ENABLE_QUERY_REWRITE = options.enable_query_rewrite;
    const int CHAMPION_LAYER_LEVEL = options.champion_layer_level;
    // -1：原始 HNSW 随机分层，并保留每类别至少 R 个冠军层节点；
    // [0,1]：level >= lc 的目标比例，同样保留每类别最低节点规则。
    const double CHAMPION_LAYER_RATIO = options.champion_layer_ratio;
    // 如还需取消“每类至少 R 个”的保底，请将 MIN_PER_CATEGORY 设为 0。
    const size_t CHAMPION_LAYER_MIN_PER_CATEGORY =
        options.champion_layer_min_per_category;
    const size_t CHAMPION_REWRITE_TOPK =
        options.champion_rewrite_topk; // 仅 query_rewrite_method=0 时生效。
    
    heuristic_control = options.heuristic_control;

    /*
     * ACORN 构建模式：
     * 0：ACORN-1（普通 HNSW 构建，查询时使用二跳扩展）
     * 1：未压缩 ACORN-gamma（邻居规模扩展为 M*gamma）
     */
    const int ACORN_BUILD_MODE = options.acorn_build_mode;

    // ACORN_BUILD_MODE == 1 uses the simplified uncompressed scheme:
    // expanded_M = M * gamma.  M_beta is fixed to expanded_M only for
    // index/CSV compatibility; no M_beta compression is performed.
    const size_t ACORN_GAMMA = options.acorn_gamma;
    const size_t ACORN_M_BETA =
        ACORN_BUILD_MODE == 1
            ? static_cast<size_t>(M) * ACORN_GAMMA
            : static_cast<size_t>(M);
    
    // ACORN-1 performs full two-hop expansion. Uncompressed ACORN-gamma
    // obtains its candidates from the expanded direct-neighbor list.
    const int hop_count = options.hop_count;
    const int init_hop_count = options.init_hop_count;
    const int query_rewrite_method = options.query_rewrite_method;
    const int rgn_rewrite_hop_count = options.rgn_rewrite_hop_count; /** RGN 在冠军层的局部邻居扩展跳数*/

    if (ACORN_BUILD_MODE != 0 && ACORN_BUILD_MODE != 1) {
        std::cerr << "ACORN_BUILD_MODE must be 0 (ACORN-1) or 1 "
                     "(ACORN-gamma)." << std::endl;
        return EXIT_FAILURE;
    }
    if (ACORN_BUILD_MODE == 1 && ACORN_GAMMA < 2) {
        std::cerr << "ACORN-gamma requires ACORN_GAMMA >= 2."
                  << std::endl;
        return EXIT_FAILURE;
    }

    const bool SAVE_INDEX_FILE = options.save_index_file;
    const bool SAVE_CSV = options.save_csv;

    // 【建议】你可以手动修改这里的文件名，把 C100 换成 L1R100 之类的，方便区分实验
    const std::string data_path = options.data_path;
    const std::string hnsw_index_path = options.hnsw_index_path;
    const std::string brute_index_path = options.brute_index_path;
    const std::string csv_save_path = options.csv_save_path;

    hnswlib::HierarchicalNSW<float>* alg_hnsw = nullptr;
    std::vector<float> global_data_buffer;

    double avg_brute_dist = 0.0;
    std::vector<double> recall, hnsw_time, search_dist, dist_opt, time_opt;
    std::vector<double> champion_layer_rewrite_dist; // 【修改27】改名

    std::vector<uint64_t> hetero_edges_per_layer;
    std::vector<uint64_t> homo_edges_per_layer;
    std::vector<double> hetero_ratio_per_layer;
    std::vector<double> effective_edge_ratio_per_layer;

    std::vector<std::unordered_map<uint16_t, int>> top_candidates_stats;
    bool all_stats_same = false;
    std::vector<double> overall_blank_rates;
    std::vector<double> average_nfr_rates;
    std::vector<std::vector<std::pair<uint16_t, double>>> blank_rates_details_list;

    std::cout << "========================================" << std::endl;
    std::cout << "Biased Category HNSW Index Evaluation Program (Champion Layer)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Initial Configuration:" << std::endl;
    std::cout << "  - Dataset: " << data_path << std::endl;
    std::cout << "  - M: " << M << std::endl;
    std::cout << "  - ef_construction: " << ef_construction << std::endl;
    std::cout << "  - k (Recall@k): " << k << std::endl;
    std::cout << "  - ef_search values: ";
    for (size_t i = 0; i < efSearchs.size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << efSearchs[i];
    }
    std::cout << std::endl;
    std::cout << "  - Query count: " << query_count << std::endl;
    std::cout << "  - Query seed: " << query_seed << std::endl;
    std::cout << "  - Index random seed: "
              << options.index_random_seed << std::endl;
    std::cout << "  - Allow replace deleted: "
              << (options.allow_replace_deleted ? "YES" : "NO") << std::endl;
    std::cout << "  - BIAS_DEGREE: " << BIAS_DEGREE << " (0=无偏差,1=最大偏差)" << std::endl;
    std::cout << "  - hop_count: " << hop_count << std::endl;
    std::cout << "  - init_hop_count: " << init_hop_count << std::endl;
    std::cout << "  - RGN rewrite hop count: "
              << rgn_rewrite_hop_count << std::endl;
    std::cout << "  - Query rewrite method: "
              << query_rewrite_method << std::endl;
    std::cout << "  - ACORN Build Mode: "
              << (ACORN_BUILD_MODE == 0
                      ? "ACORN-1"
                      : "ACORN-gamma (uncompressed)")
              << std::endl;
    std::cout << "  - ACORN gamma: "
              << (ACORN_BUILD_MODE == 0 ? 1 : ACORN_GAMMA)
              << std::endl;
    std::cout << "  - ACORN M_beta metadata (compression disabled): "
              << (ACORN_BUILD_MODE == 0 ? static_cast<size_t>(M) : ACORN_M_BETA)
              << std::endl;
    std::cout << "  - ACORN expanded degree M*gamma: "
              << (ACORN_BUILD_MODE == 0
                      ? static_cast<size_t>(M)
                      : ACORN_M_BETA)
              << std::endl;
    // 【修改28】打印配置更新
    std::cout << "  - Enable Query Rewrite: " << (ENABLE_QUERY_REWRITE ? "YES" : "NO") << std::endl;
    std::cout << "  - Champion Layer Level (lc): " << CHAMPION_LAYER_LEVEL << std::endl;
    std::cout << "  - Champion Layer Ratio: " << CHAMPION_LAYER_RATIO << std::endl;
    std::cout << "  - Champion Layer Min Per Category (R): " << CHAMPION_LAYER_MIN_PER_CATEGORY << std::endl;
    std::cout << "  - Champion Rewrite TopK: " << CHAMPION_REWRITE_TOPK << std::endl;
    std::cout << "  - Save All Index Files: " << (SAVE_INDEX_FILE ? "YES" : "NO") << std::endl;
    std::cout << "  - Save CSV Results: " << (SAVE_CSV ? "YES" : "NO") << std::endl;
    std::cout << "  - HNSW Index Path: " << hnsw_index_path << std::endl;
    std::cout << "  - Build Metrics Path: " << brute_index_path << std::endl;
    std::cout << "  - CSV Save Path: " << csv_save_path << std::endl;
    std::cout << "  - Heuristic Control: " << heuristic_control << std::endl;
    std::cout << "========================================" << std::endl;

    if (options.dry_run) {
        std::cout << " Dry run completed: configuration is valid; no data was loaded."
                  << std::endl;
        return EXIT_SUCCESS;
    }

    if (options.skip_if_csv_complete &&
        csv_has_complete_results(csv_save_path, efSearchs.size())) {
        std::cout << " Complete CSV already exists; this experiment is skipped."
                  << std::endl;
        return EXIT_SUCCESS;
    }

    try {
        if (!file_exists(data_path)) {
            std::cerr << "\n Critical Error: Dataset file not found - " << data_path << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << " Dataset file exists: " << data_path << std::endl;

        global_data_buffer = load_labeled_data(data_path);

        for (size_t i = 0; i < point_categories.size(); ++i) {
            int c = point_categories[i];
            if (c < 0 || c >= static_cast<int>(global_num_clusters)) {
                throw std::runtime_error("Found out-of-range category in labeled data file.");
            }
        }

        preprocess_category_groups(global_num_points);

        bool need_rebuild =
            !file_exists(hnsw_index_path) ||
            !file_exists(brute_index_path) ||
            !build_metrics_match_champion_layer_ratio(
                brute_index_path,
                heuristic_control,
                CHAMPION_LAYER_RATIO,
                2,
                CHAMPION_LAYER_LEVEL,
                CHAMPION_LAYER_MIN_PER_CATEGORY);
        if (need_rebuild) {
            std::cout << "\nMissing or incompatible index/build metrics -> "
                         "Rebuild HNSW index + metrics..." << std::endl;

            hnswlib::L2Space hnsw_space(static_cast<size_t>(global_dim));
            global_dist_func = hnsw_space.get_dist_func();
            global_dist_func_param = hnsw_space.get_dist_func_param();

            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                &hnsw_space,
                static_cast<size_t>(global_num_points),
                M,
                ef_construction,
                options.index_random_seed,
                options.allow_replace_deleted,
                ACORN_BUILD_MODE,
                ACORN_GAMMA,
                ACORN_M_BETA
            );

            // 【修改29】接口调用大换血：去掉 Champion List，设置 Champion Layer
            alg_hnsw->setEnableQueryRewrite(ENABLE_QUERY_REWRITE);
            alg_hnsw->setLevel0NeighborControl(heuristic_control);
            alg_hnsw->setChampionLayerLevel(CHAMPION_LAYER_LEVEL);
            alg_hnsw->setChampionLayerRatio(CHAMPION_LAYER_RATIO);
            alg_hnsw->setChampionLayerMinPerCategory(CHAMPION_LAYER_MIN_PER_CATEGORY);

            if (ENABLE_QUERY_REWRITE) {
                alg_hnsw->setChampionRewriteTopk(CHAMPION_REWRITE_TOPK);
            } else {
                alg_hnsw->setChampionRewriteTopk(0);
            }

            build_hnsw_index(
                alg_hnsw,
                global_num_points,
                static_cast<int>(global_dim),
                global_data_buffer
            );

            if (SAVE_INDEX_FILE == 1) {
                const std::string temporary_hnsw_path =
                    hnsw_index_path + ".building";
                const std::string temporary_metrics_path =
                    brute_index_path + ".building";

                alg_hnsw->saveIndex(temporary_hnsw_path);
                save_build_metrics(temporary_metrics_path);
                replace_file_from_temporary(
                    temporary_hnsw_path,
                    hnsw_index_path);
                replace_file_from_temporary(
                    temporary_metrics_path,
                    brute_index_path);
                std::cout << " HNSW index + build metrics saved successfully!" << std::endl;
            }
        } else {
            std::cout << "\nAll files exist -> Load HNSW index + build metrics directly..." << std::endl;

            load_build_metrics(brute_index_path);

            hnswlib::L2Space hnsw_space(static_cast<size_t>(global_dim));
            global_dist_func = hnsw_space.get_dist_func();
            global_dist_func_param = hnsw_space.get_dist_func_param();

            alg_hnsw = new hnswlib::HierarchicalNSW<float>(
                &hnsw_space,
                hnsw_index_path,
                false,
                0,
                options.allow_replace_deleted);

            const size_t expected_gamma =
                ACORN_BUILD_MODE == 1 ? ACORN_GAMMA : 1;
            const size_t expected_m_beta =
                ACORN_BUILD_MODE == 1
                    ? ACORN_M_BETA
                    : static_cast<size_t>(M);
            const size_t expected_ef_construction = std::max(
                static_cast<size_t>(ef_construction),
                expected_m_beta);
            if (alg_hnsw->M_ != static_cast<size_t>(M) ||
                alg_hnsw->ef_construction_ != expected_ef_construction ||
                alg_hnsw->getACORNBuildMode() != ACORN_BUILD_MODE ||
                alg_hnsw->getACORNGamma() != expected_gamma ||
                alg_hnsw->getACORNMbeta() != expected_m_beta ||
                alg_hnsw->getChampionLayerRatio() != CHAMPION_LAYER_RATIO ||
                alg_hnsw->getChampionLayerAssignmentVersion() != 2 ||
                alg_hnsw->getChampionLayerLevel() != CHAMPION_LAYER_LEVEL ||
                alg_hnsw->getChampionLayerMinPerCategory() !=
                    CHAMPION_LAYER_MIN_PER_CATEGORY) {
                throw std::runtime_error(
                    "Loaded index does not match M, ef_construction, ACORN_BUILD_MODE, "
                    "ACORN_GAMMA, ACORN_M_BETA, or champion-layer build settings. Use a separate "
                    "index path or rebuild the index.");
            }
            
            // 【修改30】加载分支接口调用同样更新
            alg_hnsw->setEnableQueryRewrite(ENABLE_QUERY_REWRITE);
            alg_hnsw->setLevel0NeighborControl(heuristic_control);
            alg_hnsw->setChampionLayerLevel(CHAMPION_LAYER_LEVEL);
            alg_hnsw->setChampionLayerRatio(CHAMPION_LAYER_RATIO);
            alg_hnsw->setChampionLayerMinPerCategory(CHAMPION_LAYER_MIN_PER_CATEGORY);

            if (ENABLE_QUERY_REWRITE) {
                alg_hnsw->setChampionRewriteTopk(CHAMPION_REWRITE_TOPK);
            } else {
                alg_hnsw->setChampionRewriteTopk(0);
            }
            std::cout << " HNSW index loaded successfully (elements: "
                      << alg_hnsw->getCurrentElementCount() << ")" << std::endl;
        }

        analyze_all_layers_topology(
            alg_hnsw, 
            hetero_edges_per_layer, 
            homo_edges_per_layer, 
            hetero_ratio_per_layer,
            effective_edge_ratio_per_layer
        );

        generate_queries_from_categories(query_count, query_seed);

        precompute_category_geometry(
            static_cast<int>(global_dim),
            global_data_buffer
        );

        size_t valid_rho_query_count = 0;
        size_t negative_rho_query_count = 0;
        const double average_query_rho = prepare_query_correlations(
            static_cast<int>(global_dim),
            BIAS_DEGREE,
            global_data_buffer,
            valid_rho_query_count,
            negative_rho_query_count
        );

        const double negative_query_ratio_percent =
            valid_rho_query_count > 0
                ? 100.0 * static_cast<double>(negative_rho_query_count)
                    / static_cast<double>(valid_rho_query_count)
                : 0.0;

        std::cout << std::fixed << std::setprecision(8);
        std::cout << " Average Filtered Query Correlation rho: "
                  << average_query_rho << std::endl;
        std::cout << " Valid rho Query Count: "
                  << valid_rho_query_count << std::endl;
        std::cout << " Negative rho Query Count: "
                  << negative_rho_query_count << std::endl;
        std::cout << " Negative Query Ratio: "
                  << std::setprecision(4) << negative_query_ratio_percent
                  << "%" << std::endl;

        precompute_brute_force_results(
            static_cast<int>(global_dim),
            k,
            BIAS_DEGREE,
            global_data_buffer
        );

        evaluate_search_performance(
            alg_hnsw,
            global_num_points,
            static_cast<int>(global_dim),
            k,
            BIAS_DEGREE,
            efSearchs,
            ef_construction,
            M,
            global_data_buffer,
            avg_brute_dist,
            recall,
            hnsw_time,
            search_dist,
            dist_opt,
            time_opt,
            champion_layer_rewrite_dist, // 【修改31】改名
            hop_count,
            init_hop_count,
            query_rewrite_method,
            rgn_rewrite_hop_count,
            top_candidates_stats,
            all_stats_same,
            overall_blank_rates,
            average_nfr_rates,
            blank_rates_details_list
        );

        // 【修改32】CSV 参数准备更新
        const int CSV_CHAMPION_LAYER_LEVEL =
            ENABLE_QUERY_REWRITE ? CHAMPION_LAYER_LEVEL : 0;

        const size_t CSV_CHAMPION_LAYER_MIN_PER_CATEGORY =
            ENABLE_QUERY_REWRITE ? CHAMPION_LAYER_MIN_PER_CATEGORY : 0;

        const size_t CSV_CHAMPION_REWRITE_TOPK =
            ENABLE_QUERY_REWRITE ? CHAMPION_REWRITE_TOPK : 0;

        if (SAVE_CSV == 1) {
            const std::string temporary_csv_path =
                csv_save_path + ".building";
            save_results_to_csv(
                temporary_csv_path,
                efSearchs,
                M,
                ef_construction,
                k,
                static_cast<int>(global_dim),
                global_num_points,
                data_path,
                query_count,
                query_seed,
                BIAS_DEGREE,
                average_query_rho,
                valid_rho_query_count,
                negative_rho_query_count,
                negative_query_ratio_percent,
                global_num_clusters,
                hop_count,
                init_hop_count,
                query_rewrite_method,
                rgn_rewrite_hop_count,
                ACORN_BUILD_MODE,
                ACORN_BUILD_MODE == 1 ? ACORN_GAMMA : 1,
                ACORN_BUILD_MODE == 1 ? ACORN_M_BETA : static_cast<size_t>(M),
                CSV_CHAMPION_LAYER_LEVEL, // 传入新参数
                CHAMPION_LAYER_RATIO,
                CSV_CHAMPION_LAYER_MIN_PER_CATEGORY, // 传入新参数
                CSV_CHAMPION_REWRITE_TOPK,
                brute_force_avg_time,
                avg_brute_dist,
                recall,
                hnsw_time,
                search_dist,
                champion_layer_rewrite_dist, // 【修改33】改名
                dist_opt,
                time_opt,
                // 【修改34】删除：传入 champion_build_dist_calc
                overall_blank_rates,
                average_nfr_rates,
                blank_rates_details_list,
                hetero_ratio_per_layer,
                effective_edge_ratio_per_layer,
                heuristic_control
            );
            replace_file_from_temporary(
                temporary_csv_path,
                csv_save_path);
            std::cout << "CSV result published successfully: "
                      << csv_save_path << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        if (alg_hnsw) delete alg_hnsw;
        return EXIT_FAILURE;
    }

    if (alg_hnsw) delete alg_hnsw;

    std::cout << "\n========================================" << std::endl;
    std::cout << " Program Completed Successfully" << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
