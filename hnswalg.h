#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <cstdint>  // 新增：64位整数支持
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cmath>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    // 0: ACORN-1; 1: uncompressed ACORN-gamma.
    int acorn_build_mode_{0};
    size_t acorn_gamma_{1};
    size_t acorn_m_beta_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    // 新增：类别字段的内存偏移量
    size_t category_offset_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    // 新增：专门统计「搜索阶段」的距离计算次数（原子变量保证多线程安全）
    mutable std::atomic<long> search_distance_computations_{0};

    // 新增：专门统计「构建阶段」的距离计算次数（原子变量保证多线程安全）
    mutable std::atomic<uint64_t> build_distance_computations_{0};

    // 搜索时候选列表中节点的数量
    mutable std::atomic<uint16_t> search_top_candidates_count_{0};

    // =============开启冠军层改写=============
    // 是否启用冠军层查询改写
    bool enable_query_rewrite_{true};

    // 冠军层层号，由用户指定，要求 >=1
    int champion_layer_level_{1};

    // 每个类别在冠军层至少保留多少个节点
    size_t champion_layer_min_per_category_{10};

    // 冠军层（level >= champion_layer_level_）节点占底层全图节点的目标比例。
    // -1 表示使用原始 HNSW 随机分层，同时保留每类别最少节点规则。
    double champion_layer_ratio_{-1.0};

    // 2：-1 兼容修改前的冠军层最低节点规则。
    uint32_t champion_layer_assignment_version_{2};

    // 查询改写时，从冠军层搜索结果里取多少个点做均值
    size_t champion_rewrite_topk_{10};

    // 记录每个类别在冠军层的“入口节点”
    // category -> internal_id
    mutable std::unordered_map<int, tableint> champion_layer_entrypoints_;

    // 记录每个类别已经被强制提升到冠军层的数量
    // 仅构建时使用
    mutable std::unordered_map<int, size_t> champion_layer_forced_count_;

    // 冠军层查询改写阶段的距离计算次数
    mutable std::atomic<uint64_t> champion_rewrite_distance_computations_{0};

    // 构建冠军层图结构时的距离计算次数
    mutable std::atomic<uint64_t> champion_layer_build_distance_computations_{0};


    // 为0保持原始heuristic，为1底层补同构边，为-1底层补异构边
    int level0_neighbor_control_ = 1;

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false,
        int acorn_build_mode = 0,
        size_t acorn_gamma = 1,
        size_t acorn_m_beta = 0)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        if (acorn_build_mode != 0 && acorn_build_mode != 1) {
            throw std::invalid_argument("acorn_build_mode must be 0 (ACORN-1) or 1 (ACORN-gamma)");
        }
        if (acorn_build_mode == 1 && acorn_gamma < 2) {
            throw std::invalid_argument("ACORN-gamma requires gamma >= 2");
        }

        acorn_build_mode_ = acorn_build_mode;
        acorn_gamma_ = acorn_build_mode_ == 1 ? acorn_gamma : 1;
        // The simplified ACORN-gamma implementation is deliberately
        // uncompressed.  M_beta is retained in the public interface and the
        // index metadata for backward-compatible experiment logging, but in
        // this mode its only valid value is M*gamma (or 0 for auto-select).
        const size_t expanded_M = M_ * acorn_gamma_;
        if (acorn_build_mode_ == 1 &&
            acorn_m_beta != 0 &&
            acorn_m_beta != expanded_M) {
            throw std::invalid_argument(
                "Uncompressed ACORN-gamma requires M_beta == M * gamma");
        }
        acorn_m_beta_ = acorn_build_mode_ == 1 ? expanded_M : M_;

        const size_t max_storable_neighbors =
            static_cast<size_t>(std::numeric_limits<unsigned short>::max());
        if (M_ > max_storable_neighbors / (2 * acorn_gamma_)) {
            throw std::invalid_argument("2 * M * gamma exceeds link-list count capacity");
        }
        // ACORN-1 uses M/2M link-list capacities.  Uncompressed
        // ACORN-gamma replaces the structural degree M by M*gamma while
        // retaining the original M for level generation and query-time
        // post-filter traversal limits.
        maxM_ = expanded_M;
        maxM0_ = 2 * expanded_M;

        // The construction beam must be able to return at least M*gamma
        // candidates for the uncompressed neighbor list.
        const size_t construction_candidate_budget = expanded_M;
        ef_construction_ = std::max(
            ef_construction,
            construction_candidate_budget);
        ef_ = 10;
        
        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

        // 修改后（加入int型类别字段，后面的int表示这个节点的类别
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype) + sizeof(int);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        category_offset_ = label_offset_ + sizeof(labeltype); // 类别字段偏移量
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }

    // 获取最高层编号（从0开始）
    int getMaxLevel() const {
        return maxlevel_;
    }

    void setEnableQueryRewrite(bool flag) {
        enable_query_rewrite_ = flag;
    }

    int getACORNBuildMode() const {
        return acorn_build_mode_;
    }

    size_t getACORNGamma() const {
        return acorn_gamma_;
    }

    size_t getACORNMbeta() const {
        return acorn_m_beta_;
    }

    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    /*return_label 是外部的标签，internal_id是内部标签*/
    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        // std::cout << "\n internal_id: " << internal_id << " return_label: " << return_label; 
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    void setChampionLayerLevel(int lc) {
        if (lc < 1) {
            throw std::runtime_error("champion layer level must be >= 1");
        }
        champion_layer_level_ = lc;
    }

    int getChampionLayerLevel() const {
        return champion_layer_level_;
    }

    void setChampionLayerMinPerCategory(size_t R) {
        champion_layer_min_per_category_ = R;
    }

    size_t getChampionLayerMinPerCategory() const {
        return champion_layer_min_per_category_;
    }

    void setChampionLayerRatio(double ratio) {
        if (!std::isfinite(ratio) ||
            (ratio != -1.0 && (ratio < 0.0 || ratio > 1.0))) {
            throw std::invalid_argument(
                "champion layer ratio must be -1 or in [0, 1]");
        }
        champion_layer_ratio_ = ratio;
    }

    double getChampionLayerRatio() const {
        return champion_layer_ratio_;
    }

    uint32_t getChampionLayerAssignmentVersion() const {
        return champion_layer_assignment_version_;
    }

    void setChampionRewriteTopk(size_t topk) {
        champion_rewrite_topk_ = topk;
    }

    size_t getChampionRewriteTopk() const {
        return champion_rewrite_topk_;
    }

    bool hasChampionLayerEntrypoint(int category) const {
        return champion_layer_entrypoints_.find(category) != champion_layer_entrypoints_.end();
    }

    tableint getChampionLayerEntrypoint(int category) const {
        auto it = champion_layer_entrypoints_.find(category);
        if (it == champion_layer_entrypoints_.end()) {
            throw std::runtime_error("No champion-layer entrypoint for this category");
        }
        return it->second;
    }

    uint64_t getChampionRewriteDistanceComputations() const {
        return champion_rewrite_distance_computations_.load();
    }

    void resetChampionRewriteDistanceComputations() {
        champion_rewrite_distance_computations_.store(0);
    }

    // 新增：根据内部ID获取向量类别
    inline int getCategoryByInternalId(tableint internal_id) const {
        int category;
        memcpy(&category, (data_level0_memory_ + internal_id * size_data_per_element_ + category_offset_), sizeof(int));
        return category;
    }

    // 新增：设置内部ID对应的向量类别
    inline void setCategoryByInternalId(tableint internal_id, int category) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + category_offset_), &category, sizeof(int));
    }

    // （可选）获取类别字段指针，方便批量操作
    inline int *getCategoryPtrByInternalId(tableint internal_id) const {
        return (int *) (data_level0_memory_ + internal_id * size_data_per_element_ + category_offset_);
    }

    // 新增：获取搜索阶段的距离计算总次数
    long getSearchDistanceComputations() const {
        return search_distance_computations_.load(); // 原子加载保证线程安全
    }

    // 新增：重置搜索距离计算次数统计（比如每次搜索前清零）
    void resetSearchDistanceComputations() {
        search_distance_computations_.store(0); // 原子存储保证线程安全
    }

    // 新增：获取构建阶段的距离计算总次数
    uint64_t getBuildDistanceComputations() const {
        return build_distance_computations_.load(); // 原子加载保证线程安全
    }

    // 新增：重置构建距离计算次数统计（比如每次构建前清零）
    void resetBuildDistanceComputations() {
        build_distance_computations_.store(0); // 原子存储保证线程安全
    }

    uint64_t getChampionLayerBuildDistanceComputations() const {
        return champion_layer_build_distance_computations_.load();
    }

    void resetChampionLayerBuildDistanceComputations() {
        champion_layer_build_distance_computations_.store(0);
    }

    uint16_t getSearchTopCandidatesCount() const {
        return search_top_candidates_count_.load();
    }

    void resetSearchTopCandidatesCount() {
        search_top_candidates_count_.store(0);
    }

    inline size_t getVectorDim() const {
        return *((size_t*)dist_func_param_);
    }

    inline dist_t calcDistanceByInternalId(tableint a, tableint b) const {
        dist_t d = fstdistfunc_(getDataByInternalId(a), getDataByInternalId(b), dist_func_param_);
        return d;
    }    

    void setLevel0NeighborControl(int control) {
        if (control != -1 && control != 0 && control != 1) {
            throw std::runtime_error("level0 neighbor control must be one of {-1, 0, 1}");
        }
        level0_neighbor_control_ = control;
    }

    double getHeterogeneousEdgeRatioAtLayer(
        int layer,
        uint64_t& heterogeneous_edges,
        uint64_t& homogeneous_edges
    ) const {
        if (layer < 0) {
            throw std::runtime_error("layer must be >= 0");
        }

        heterogeneous_edges = 0;
        homogeneous_edges = 0;

        const size_t cur_count = cur_element_count.load();

        for (tableint node_id = 0; node_id < cur_count; ++node_id) {
            if (element_levels_[node_id] < layer) {
                continue;
            }

            if (isMarkedDeleted(node_id)) {
                continue;
            }

            const int node_category = getCategoryByInternalId(node_id);

            linklistsizeint* ll_cur = get_linklist_at_level(node_id, layer);
            const unsigned short neighbor_count = getListCount(ll_cur);
            tableint* neighbors = (tableint*)(ll_cur + 1);

            for (unsigned short j = 0; j < neighbor_count; ++j) {
                tableint neighbor_id = neighbors[j];

                if (neighbor_id >= cur_count) {
                    continue;
                }

                if (isMarkedDeleted(neighbor_id)) {
                    continue;
                }

                const int neighbor_category = getCategoryByInternalId(neighbor_id);

                if (neighbor_category == node_category) {
                    homogeneous_edges++;
                } else {
                    heterogeneous_edges++;
                }
            }
        }

        const uint64_t total_edges = heterogeneous_edges + homogeneous_edges;
        if (total_edges == 0) {
            return 0.0;
        }

        return static_cast<double>(heterogeneous_edges) /
            static_cast<double>(total_edges);
    }

    /*该searchBaseLayer构建索引时专用，搜索时不用*/
    /*构建索引时，上层查找最近邻，建立连接，不进行属性过滤，与最近的M个节点建立连接，与原版一致*/
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            build_distance_computations_++;   // 新增：统计构建距离计算次数
            if (layer == champion_layer_level_) {
                champion_layer_build_distance_computations_++;
            }
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound &&
                top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                build_distance_computations_++;   // 新增：统计构建距离计算次数
                if (layer == champion_layer_level_) {
                    champion_layer_build_distance_computations_++;
                }
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    // 注意，目前查询改写的距离计算次数会算到搜索时距离计算的次数上
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer_WithCategory(
        tableint ep_id,
        const void *data_point,
        int layer,
        int query_category,
        size_t ef,
        int hop_count,        // 正式搜索时的hop扩展
        int init_hop_count    // 初始热启动扩展跳数
    ){
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        using CandidateQueue = std::priority_queue<
            std::pair<dist_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            CompareByFirst>;

        CandidateQueue top_candidates;
        CandidateQueue candidate_set;
        dist_t lowerBound = std::numeric_limits<dist_t>::max();

        // =============================
        // 初始播种跳数：仅用于正式搜索前
        // 和后续搜索的 hop_count 不是一个东西
        // =============================

        auto try_add_candidate = [&](tableint nid, dist_t d) {
            if (top_candidates.size() < ef || d < lowerBound) {
                candidate_set.emplace(-d, nid);
                top_candidates.emplace(d, nid);
                while (top_candidates.size() > ef) {
                    top_candidates.pop();
                }
                lowerBound = top_candidates.top().first;
            }
        };

        // 入口点至少要进入 candidate_set，保证后续搜索能从它开始
        candidate_set.emplace(-std::numeric_limits<dist_t>::max(), ep_id);
        visited_array[ep_id] = visited_array_tag;

        // ==================================================
        // 阶段1：初始 k 跳播种（只在 layer==0 做）
        // 目的：先把入口附近同类别点放进 top_candidates / candidate_set
        // ==================================================
        if (layer == 0) {
            std::queue<std::pair<tableint, int>> init_queue;
            init_queue.emplace(ep_id, 0);

            while (!init_queue.empty()) {
                auto [node_id, hop] = init_queue.front();
                init_queue.pop();

                // 当前点如果类别匹配，则作为“初始种子”加入候选集
                if (!isMarkedDeleted(node_id) &&
                    getCategoryByInternalId(node_id) == query_category) {
                    const dist_t d = fstdistfunc_(
                        data_point,
                        getDataByInternalId(node_id),
                        dist_func_param_);
                    search_distance_computations_++;
                    try_add_candidate(node_id, d);
                }

                // 继续向外扩 init_hop_cout 跳
                if (hop >= init_hop_count) {
                    continue;
                }

                int* node_neighbors = (int*)get_linklist0(node_id);
                size_t node_neighbor_count = getListCount((linklistsizeint*)node_neighbors);

                for (size_t j = 1; j <= node_neighbor_count; ++j) {
                    const tableint next_id = *(node_neighbors + j);
                    if (next_id >= max_elements_ ||
                        visited_array[next_id] == visited_array_tag) {
                        continue;
                    }

                    visited_array[next_id] = visited_array_tag;
                    init_queue.emplace(next_id, hop + 1);
                }
            }
        } else {
            // 非0层保持原先入口点逻辑
            const bool ep_valid =
                !isMarkedDeleted(ep_id) &&
                (getCategoryByInternalId(ep_id) == query_category);

            if (ep_valid) {
                const dist_t dist = fstdistfunc_(
                    data_point,
                    getDataByInternalId(ep_id),
                    dist_func_param_);
                search_distance_computations_++;
                champion_rewrite_distance_computations_++;
                try_add_candidate(ep_id, dist);
            }
        }

        const size_t TOP_KEEP =
            acorn_build_mode_ == 1 ? 2 * M_ : maxM0_;
        std::vector<std::pair<dist_t, tableint>> temp_neighbors;

        // ==================================================
        // 阶段2：正式搜索
        // ==================================================
        while (!candidate_set.empty()) {
            auto curr = candidate_set.top();
            candidate_set.pop();
            const dist_t curr_dist = -curr.first;
            const tableint curr_id = curr.second;

            if (curr_dist > lowerBound && top_candidates.size() >= ef) {
                break;
            }

            int* neighbors = (layer == 0)
                ? (int*)get_linklist0(curr_id)
                : (int*)get_linklist(curr_id, layer);
            const size_t neighbor_count = getListCount((linklistsizeint*)neighbors);

            if (layer > 0) {
                const size_t upper_neighbor_limit = neighbor_count;
                for (size_t j = 1; j <= upper_neighbor_limit; ++j) {
                    const tableint nid = *(neighbors + j);
                    if (nid >= max_elements_ || visited_array[nid] == visited_array_tag) {
                        continue;
                    }
                    visited_array[nid] = visited_array_tag;

                    if (!isMarkedDeleted(nid) &&
                        getCategoryByInternalId(nid) == query_category) {
                        const dist_t d = fstdistfunc_(
                            data_point,
                            getDataByInternalId(nid),
                            dist_func_param_);
                        search_distance_computations_++;
                        champion_rewrite_distance_computations_++;
                        try_add_candidate(nid, d);
                    }
                }
                continue;
            }

            // --------------------------
            // 底层0层：正式搜索时的 k 跳邻居扩展
            // 注意：这是你原来的 hop_count，
            // 和 init_hop_count 的“初始播种”不同
            // --------------------------
            temp_neighbors.clear();
            std::queue<std::pair<tableint, int>> bfs_queue;

            // ACORN-gamma scans its complete uncompressed direct-neighbor
            // list.  Predicate filtering is applied before TOP_KEEP, whose
            // expected qualifying size is about 2*M when gamma approximates
            // the reciprocal of the minimum selectivity.
            const size_t direct_neighbor_limit = neighbor_count;

            for (size_t j = 1; j <= direct_neighbor_limit; ++j) {
                const tableint nid = *(neighbors + j);
                if (nid >= max_elements_ || visited_array[nid] == visited_array_tag) {
                    continue;
                }
                visited_array[nid] = visited_array_tag;
                bfs_queue.emplace(nid, 1);
            }

            while (!bfs_queue.empty()) {
                auto [node_id, hop] = bfs_queue.front();
                bfs_queue.pop();

                if (!isMarkedDeleted(node_id) &&
                    getCategoryByInternalId(node_id) == query_category) {
                    const dist_t d = fstdistfunc_(
                        data_point,
                        getDataByInternalId(node_id),
                        dist_func_param_);
                    search_distance_computations_++;
                    temp_neighbors.emplace_back(d, node_id);
                }

                if (acorn_build_mode_ == 0 && hop < hop_count) {
                    int* node_neighbors = (int*)get_linklist0(node_id);
                    size_t node_neighbor_count =
                        getListCount((linklistsizeint*)node_neighbors);

                    for (size_t k = 1; k <= node_neighbor_count; ++k) {
                        const tableint next_id = *(node_neighbors + k);
                        if (next_id >= max_elements_ ||
                            visited_array[next_id] == visited_array_tag) {
                            continue;
                        }
                        visited_array[next_id] = visited_array_tag;
                        bfs_queue.emplace(next_id, hop + 1);
                    }
                }
            }

            std::sort(temp_neighbors.begin(), temp_neighbors.end());

            const size_t keep = std::min(TOP_KEEP, temp_neighbors.size());

            for (size_t m = 0; m < keep; ++m) {
                const auto& [d, nid] = temp_neighbors[m];
                try_add_candidate(nid, d);
            }
        }

        visited_list_pool_->releaseVisitedList(vl);

        search_top_candidates_count_.store(
            static_cast<uint16_t>(top_candidates.size())
        );
        return top_candidates;
    }

    // 基本上没用，只在stop_condition中出现，而且这个stop_condition还没调用，此处没有任何修改，与原版一致
    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    void getNeighborsByHeuristic2(
            std::priority_queue<
                std::pair<dist_t, tableint>,
                std::vector<std::pair<dist_t, tableint>>,
                CompareByFirst> &top_candidates,
            const size_t M,
            tableint base_node_id,
            int control_mode = 0,
            int construction_layer = -1)
    {
        if (top_candidates.size() <= M && control_mode == 0) {
            return;
        }

        // 把 top_candidates 转成按“离 query/base node 更近的优先”弹出的队列
        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;

        while (!top_candidates.empty()) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        std::vector<std::pair<dist_t, tableint>> return_list;
        return_list.reserve(M);

        // control_mode != 0 时，缓存“可复活”的类别候选
        std::vector<std::pair<dist_t, tableint>> preferred_backup;
        preferred_backup.reserve(M);

        int base_category = -1;
        if (control_mode != 0) {
            base_category = getCategoryByInternalId(base_node_id);
        }

        // =====================================================
        // 第一阶段：
        // 原始 HNSW heuristic + 类别感知复活规则
        // =====================================================
        while (!queue_closest.empty()) {
            if (return_list.size() >= M) {
                break;
            }

            std::pair<dist_t, tableint> current_pair = queue_closest.top();
            dist_t dist_to_query = -current_pair.first;
            queue_closest.pop();

            bool good = true;
            tableint eliminator_id = (tableint)(-1);  // 记录“第一个淘汰 current_pair 的节点”

            // 原始 HNSW heuristic：diversity pruning
            for (const auto &selected_pair : return_list) {
                dist_t curdist =
                    fstdistfunc_(getDataByInternalId(selected_pair.second),
                                getDataByInternalId(current_pair.second),
                                dist_func_param_);
                build_distance_computations_++;  // 构建阶段距离计算次数
                if (construction_layer == champion_layer_level_) {
                    champion_layer_build_distance_computations_++;
                }

                // 若已选邻居 selected_pair 比 base_node 更“接近/覆盖” current_pair，
                // 则 current_pair 被 selected_pair 淘汰
                if (curdist < dist_to_query) {
                    good = false;
                    eliminator_id = selected_pair.second;
                    break;
                }
            }

            if (good) {
                if (return_list.size() < M) {
                    return_list.push_back(current_pair);
                }
                continue;
            }

            // 被淘汰后，如果 control_mode != 0，则按“类别感知复活规则”决定是否进入 preferred_backup
            if (control_mode != 0 && preferred_backup.size() < M) {
                const int candidate_category = getCategoryByInternalId(current_pair.second);

                bool candidate_match = false;
                if (control_mode == 1) {
                    // 偏向同构边：candidate 与 base 同类
                    candidate_match = (candidate_category == base_category);
                } 

                if (candidate_match && eliminator_id != (tableint)(-1)) {
                    const int eliminator_category = getCategoryByInternalId(eliminator_id);

                    // 只有当“淘汰者”和 base_node 不同类时，才允许 current_pair 进入复活池
                    if (eliminator_category != base_category) {
                        preferred_backup.push_back(current_pair);
                    }
                }
            }
        }

        // =====================================================
        // 第二阶段：
        // 对 preferred_backup 做“严格版”轻量 heuristic
        // 只保留通过 preferred 内部稀疏性检测的点，不再做 fallback 追加
        // =====================================================
        if (control_mode != 0 && return_list.size() < M && !preferred_backup.empty()) {
            std::vector<std::pair<dist_t, tableint>> preferred_selected;
            preferred_selected.reserve(M - return_list.size());

            // preferred_backup 本身已按 dist_to_query 从近到远进入
            for (const auto &candidate : preferred_backup) {
                if (return_list.size() + preferred_selected.size() >= M) {
                    break;
                }

                const dist_t dist_to_query = -candidate.first;
                bool good_preferred = true;

                // 只和已经选中的 preferred 节点比较，
                // 保证 preferred_backup 内部的分散性
                for (const auto &selected_preferred : preferred_selected) {
                    dist_t curdist =
                        fstdistfunc_(getDataByInternalId(selected_preferred.second),
                                    getDataByInternalId(candidate.second),
                                    dist_func_param_);
                    build_distance_computations_++;  // 构建阶段距离计算次数
                    if (construction_layer == champion_layer_level_) {
                        champion_layer_build_distance_computations_++;
                    }

                    // 如果 candidate 被已选 preferred 节点覆盖，则不再加入
                    if (curdist < dist_to_query) {
                        good_preferred = false;
                        break;
                    }
                }

                if (good_preferred) {
                    preferred_selected.push_back(candidate);
                }
            }

            // 合并进最终返回列表
            for (const auto &candidate : preferred_selected) {
                if (return_list.size() >= M) {
                    break;
                }
                return_list.push_back(candidate);
            }
        }

        // 写回 top_candidates，保持和原函数一致的输出形式
        for (const auto &current_pair : return_list) {
            top_candidates.emplace(-current_pair.first, current_pair.second);
        }
    }

    void getNearestNeighborsOrdered(
        std::priority_queue<
            std::pair<dist_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            CompareByFirst>& top_candidates,
        size_t max_neighbors)
    {
        std::vector<std::pair<dist_t, tableint>> ordered;
        ordered.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            ordered.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::sort(
            ordered.begin(),
            ordered.end(),
            [](const std::pair<dist_t, tableint>& a,
               const std::pair<dist_t, tableint>& b) {
                return a.first < b.first;
            });
        if (ordered.size() > max_neighbors) {
            ordered.resize(max_neighbors);
        }
        for (auto it = ordered.rbegin(); it != ordered.rend(); ++it) {
            top_candidates.emplace(-it->first, it->second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        const size_t selected_neighbor_limit =
            acorn_build_mode_ == 1 ? maxM_ : M_;

        if (acorn_build_mode_ == 1) {
            if (level == 0 && level0_neighbor_control_ != 0) {
                // Keep the existing category-aware bottom-layer edge
                // supplementation available to the RGN-ACORN-gamma variant.
                getNeighborsByHeuristic2(
                    top_candidates,
                    selected_neighbor_limit,
                    cur_c,
                    level0_neighbor_control_,
                    level);
            } else {
                // Pure uncompressed ACORN-gamma: retain the nearest M*gamma
                // outgoing neighbors without M_beta compression.
                getNearestNeighborsOrdered(
                    top_candidates,
                    selected_neighbor_limit);
            }
        } else {
            int heuristic_control =
                ((level == 0 || level == champion_layer_level_)
                    ? level0_neighbor_control_
                    : 0);
            getNeighborsByHeuristic2(
                top_candidates,
                M_,
                cur_c,
                heuristic_control,
                level);
        }
        
        if (top_candidates.size() > selected_neighbor_limit)
            throw std::runtime_error("Too many candidates returned by neighbor selection");
        
        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(selected_neighbor_limit);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point =
            acorn_build_mode_ == 1
                ? selectedNeighbors.front()
                : selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    build_distance_computations_++;  // 新增：统计构建阶段距离计算次数
                    if (level == champion_layer_level_) {
                        champion_layer_build_distance_computations_++;
                    }
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                        build_distance_computations_++;  // 新增：统计构建阶段距离计算次数
                        if (level == champion_layer_level_) {
                            champion_layer_build_distance_computations_++;
                        }
                    }

                    if (acorn_build_mode_ == 1) {
                        if (level == 0 && level0_neighbor_control_ != 0) {
                            getNeighborsByHeuristic2(
                                candidates,
                                Mcurmax,
                                selectedNeighbors[idx],
                                level0_neighbor_control_,
                                level);
                        } else {
                            getNearestNeighborsOrdered(
                                candidates,
                                Mcurmax);
                        }
                    } else {
                        int heuristic_control =
                            ((level == 0 || level == champion_layer_level_)
                                ? level0_neighbor_control_
                                : 0);
                        getNeighborsByHeuristic2(
                            candidates,
                            Mcurmax,
                            selectedNeighbors[idx],
                            heuristic_control,
                            level);
                    }
                    
                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    // MinGW/libstdc++ 8 can fail to return from a single stream read/write
    // whose byte count exceeds 2 GiB.  Large vector indexes (for example,
    // 1,000,448 x 512-dimensional LAION data) cross that boundary in the
    // contiguous level-0 block.  Keep every individual I/O request well below
    // the signed 32-bit limit while preserving the on-disk index format.
    static void writeIndexBlockInChunks(
        std::ostream& output,
        const char* data,
        size_t byte_count,
        const char* block_name
    ) {
        const size_t chunk_limit = 64ULL * 1024ULL * 1024ULL;
        size_t offset = 0;
        while (offset < byte_count) {
            const size_t chunk_size =
                std::min(chunk_limit, byte_count - offset);
            output.write(
                data + offset,
                static_cast<std::streamsize>(chunk_size));
            if (!output) {
                throw std::runtime_error(
                    std::string("Failed to write index block: ") + block_name);
            }
            offset += chunk_size;
        }
    }

    static void readIndexBlockInChunks(
        std::istream& input,
        char* data,
        size_t byte_count,
        const char* block_name
    ) {
        const size_t chunk_limit = 64ULL * 1024ULL * 1024ULL;
        size_t offset = 0;
        while (offset < byte_count) {
            const size_t chunk_size =
                std::min(chunk_limit, byte_count - offset);
            input.read(
                data + offset,
                static_cast<std::streamsize>(chunk_size));
            if (!input) {
                throw std::runtime_error(
                    std::string("Failed to read index block: ") + block_name);
            }
            offset += chunk_size;
        }
    }

    size_t level0DataByteCount() const {
        const size_t element_count = cur_element_count.load();
        if (size_data_per_element_ != 0 &&
            element_count >
                std::numeric_limits<size_t>::max() / size_data_per_element_) {
            throw std::runtime_error("Level-0 index byte count overflow");
        }
        return element_count * size_data_per_element_;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        // 新增：加入类别偏移量的大小
        size += sizeof(category_offset_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }

        size += sizeof(champion_layer_level_);
        size += sizeof(champion_layer_min_per_category_);
        size += sizeof(champion_rewrite_topk_);

        size_t entrypoint_count = champion_layer_entrypoints_.size();
        size += sizeof(entrypoint_count);

        for (const auto& kv : champion_layer_entrypoints_) {
            size += sizeof(int);       // category
            size += sizeof(tableint);  // entrypoint id
        }

        size += sizeof(uint32_t);  // ACORN configuration magic
        size += sizeof(acorn_build_mode_);
        size += sizeof(acorn_gamma_);
        size += sizeof(acorn_m_beta_);
        size += sizeof(uint32_t);  // champion-layer ratio magic
        size += sizeof(champion_layer_ratio_);
        size += sizeof(uint32_t);  // champion-layer assignment version magic
        size += sizeof(champion_layer_assignment_version_);

        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("Cannot open index file for writing: " + location);
        }
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        // 新增：保存类别字段偏移量
        writeBinaryPOD(output, category_offset_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        writeIndexBlockInChunks(
            output,
            data_level0_memory_,
            level0DataByteCount(),
            "level-0 data");

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                writeIndexBlockInChunks(
                    output,
                    linkLists_[i],
                    linkListSize,
                    "upper-layer links");
        }

        writeBinaryPOD(output, champion_layer_level_);
        writeBinaryPOD(output, champion_layer_min_per_category_);
        writeBinaryPOD(output, champion_rewrite_topk_);

        size_t entrypoint_count = champion_layer_entrypoints_.size();
        writeBinaryPOD(output, entrypoint_count);

        for (const auto& kv : champion_layer_entrypoints_) {
            writeBinaryPOD(output, kv.first);
            writeBinaryPOD(output, kv.second);
        }

        const uint32_t acorn_config_magic = 0x41434F52U;  // "ACOR"
        writeBinaryPOD(output, acorn_config_magic);
        writeBinaryPOD(output, acorn_build_mode_);
        writeBinaryPOD(output, acorn_gamma_);
        writeBinaryPOD(output, acorn_m_beta_);

        const uint32_t champion_ratio_magic = 0x43505231U;  // "CPR1"
        writeBinaryPOD(output, champion_ratio_magic);
        writeBinaryPOD(output, champion_layer_ratio_);

        const uint32_t champion_assignment_magic = 0x43505632U;  // "CPV2"
        writeBinaryPOD(output, champion_assignment_magic);
        writeBinaryPOD(output, champion_layer_assignment_version_);

        output.flush();
        if (!output) {
            throw std::runtime_error("Failed to flush index file: " + location);
        }
        output.close();
        if (!output) {
            throw std::runtime_error("Failed to close index file: " + location);
        }
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        champion_layer_ratio_ = -1.0;
        champion_layer_assignment_version_ = 2;
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        // 新增：读取类别字段偏移量
        readBinaryPOD(input, category_offset_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        if (M_ == 0 || maxM_ == 0 || maxM_ % M_ != 0) {
            throw std::runtime_error("Invalid ACORN index capacity metadata");
        }
        acorn_gamma_ = maxM_ / M_;
        acorn_build_mode_ = acorn_gamma_ > 1 ? 1 : 0;
        acorn_m_beta_ = acorn_build_mode_ == 1 ? maxM_ : M_;

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // 允许后面还存在扩展字段（例如冠军列表）
        // 只有越界才认为文件损坏
        if (input.tellg() > total_filesize || input.tellg() < 0)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        readIndexBlockInChunks(
            input,
            data_level0_memory_,
            level0DataByteCount(),
            "level-0 data");

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                readIndexBlockInChunks(
                    input,
                    linkLists_[i],
                    linkListSize,
                    "upper-layer links");
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        if (!input) {
            throw std::runtime_error("Index seems to be corrupted or unsupported");
        }
        if (input.peek() != EOF) {
            readBinaryPOD(input, champion_layer_level_);
            readBinaryPOD(input, champion_layer_min_per_category_);
            readBinaryPOD(input, champion_rewrite_topk_);

            size_t entrypoint_count = 0;
            readBinaryPOD(input, entrypoint_count);

            champion_layer_entrypoints_.clear();
            for (size_t i = 0; i < entrypoint_count; ++i) {
                int category;
                tableint entry_id;
                readBinaryPOD(input, category);
                readBinaryPOD(input, entry_id);
                champion_layer_entrypoints_[category] = entry_id;
            }

            if (input.peek() != EOF) {
                uint32_t acorn_config_magic = 0;
                readBinaryPOD(input, acorn_config_magic);
                if (acorn_config_magic != 0x41434F52U) {
                    throw std::runtime_error("Invalid ACORN configuration block");
                }
                readBinaryPOD(input, acorn_build_mode_);
                readBinaryPOD(input, acorn_gamma_);
                readBinaryPOD(input, acorn_m_beta_);

                const bool invalid_mode_parameters =
                    (acorn_build_mode_ == 0 &&
                        (acorn_gamma_ != 1 || acorn_m_beta_ != M_)) ||
                    (acorn_build_mode_ == 1 &&
                        (acorn_gamma_ < 2 ||
                         acorn_m_beta_ != M_ * acorn_gamma_));
                if ((acorn_build_mode_ != 0 && acorn_build_mode_ != 1) ||
                    invalid_mode_parameters ||
                    maxM_ != M_ * acorn_gamma_ ||
                    maxM0_ != 2 * maxM_) {
                    throw std::runtime_error("Invalid saved ACORN parameters");
                }
            }
        }
        if (input.peek() != EOF) {
            uint32_t champion_ratio_magic = 0;
            readBinaryPOD(input, champion_ratio_magic);
            if (champion_ratio_magic != 0x43505231U) {
                throw std::runtime_error("Invalid champion-layer ratio block");
            }
            readBinaryPOD(input, champion_layer_ratio_);
            if (!std::isfinite(champion_layer_ratio_) ||
                (champion_layer_ratio_ != -1.0 &&
                    (champion_layer_ratio_ < 0.0 || champion_layer_ratio_ > 1.0))) {
                throw std::runtime_error("Invalid saved champion-layer ratio");
            }

            // CPR1-only indexes were built by the version that skipped the
            // per-category minimum when ratio == -1.
            champion_layer_assignment_version_ = 1;
        }
        if (input.peek() != EOF) {
            uint32_t champion_assignment_magic = 0;
            readBinaryPOD(input, champion_assignment_magic);
            if (champion_assignment_magic != 0x43505632U) {
                throw std::runtime_error(
                    "Invalid champion-layer assignment version block");
            }
            readBinaryPOD(input, champion_layer_assignment_version_);
            if (champion_layer_assignment_version_ != 2) {
                throw std::runtime_error(
                    "Unsupported champion-layer assignment version");
            }
        }
        if (maxM_ != M_ * acorn_gamma_ || maxM0_ != 2 * maxM_ ||
            (acorn_build_mode_ == 1 && acorn_m_beta_ != maxM_)) {
            throw std::runtime_error(
                "This build only supports ACORN-1 or uncompressed ACORN-gamma indexes");
        }
        // 最终应该正好读到文件末尾
        if (input.peek() != EOF) {
            throw std::runtime_error("Index seems to be corrupted or unsupported");
        }
        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    // 对外接口：带 category 参数的 addPoint
    void addPoint(const void *data, labeltype label, int category, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data, label, category, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data, label, category, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);
            // 修复：复用删除节点时，必须设置类别
            setCategoryByInternalId(internal_id_replaced, category);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                // 注意，这里的updatepoint函数也在addpoint函数中用上了，他也调用了getNeighbors这个函数
                // 我们还是按照原来的套路。底层的最大值也设置为M_吧，和前面保持统一
                int heuristic_control = ((layer == 0 || layer == champion_layer_level_) ? level0_neighbor_control_ : 0);
                getNeighborsByHeuristic2(candidates,
                                        maxM_,
                                        neigh,
                                        heuristic_control,
                                        layer);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }

    // 兼容原有接口，避免抽象类
    void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) override {
        addPoint(datapoint, label, 0, replace_deleted);
    }

    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnn(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override {
        throw std::runtime_error("Please use searchKnn with category parameter");
    }

    int decideLevelForNewPoint(int category, int user_level = -1) {
        int curlevel = 0;

        if (champion_layer_ratio_ < 0.0) {
            // -1：严格保留原始 HNSW 的随机层数分配。
            curlevel = getRandomLevel(mult_);
        } else {
            // 先按目标比例决定节点是否属于冠军层，再分别使用原始
            // HNSW 分布在冠军层上下采样。这样 P(level >= lc) 等于
            // champion_layer_ratio_，同时保留层内的几何分布。
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            const bool belongs_to_champion_layer =
                distribution(level_generator_) < champion_layer_ratio_;

            if (belongs_to_champion_layer) {
                curlevel = champion_layer_level_ + getRandomLevel(mult_);
            } else {
                do {
                    curlevel = getRandomLevel(mult_);
                } while (curlevel >= champion_layer_level_);
            }
        }

        if (user_level > 0) {
            curlevel = user_level;
        }

        size_t forced_count = 0;
        auto it = champion_layer_forced_count_.find(category);
        if (it != champion_layer_forced_count_.end()) {
            forced_count = it->second;
        }

        if (forced_count < champion_layer_min_per_category_ &&
            curlevel < champion_layer_level_) {
            curlevel = champion_layer_level_;
        }

        return curlevel;
    }

    void registerChampionLayerNode(tableint internal_id, int category, int level) {
        if (level < champion_layer_level_) {
            return;
        }

        auto ep_it = champion_layer_entrypoints_.find(category);
        if (ep_it == champion_layer_entrypoints_.end()) {
            champion_layer_entrypoints_[category] = internal_id;
        }

        size_t &cnt = champion_layer_forced_count_[category];
        if (cnt < champion_layer_min_per_category_) {
            cnt++;
        }
    }


    // ===================== 3. addPoint 全3参数调用（无层级报错） =====================
    tableint addPoint(const void *data_point, labeltype label, int category, int level) {

        tableint cur_c = 0;
        {
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                setCategoryByInternalId(existingInternalId, category);
                updatePoint(data_point, existingInternalId, 1.0);
                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = decideLevelForNewPoint(category, level);

        registerChampionLayerNode(cur_c, category, curlevel);

        // 🔴 修复：添加这一行，存储节点的实际层级！
        element_levels_[cur_c] = curlevel;  

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        setCategoryByInternalId(cur_c, category); 

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                build_distance_computations_++;  // 新增：统计构建阶段距离计算次数
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);
                        if (acorn_build_mode_ == 1) {
                            size = static_cast<int>(
                                std::min(static_cast<size_t>(size), M_));
                        }
                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            build_distance_computations_++;  // 新增：统计构建阶段距离计算次数
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                // 🔴 绝对3参数调用，无多余参数
                auto top_candidates = searchBaseLayer(currObj, data_point, level);

                // std::cout << "[addPoint] 层级 " << level << " | 候选节点数量(top_candidates): " << top_candidates.size() << std::endl;

                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    build_distance_computations_++;  // 新增：统计构建阶段距离计算次数
                    if (level == champion_layer_level_) {
                        champion_layer_build_distance_computations_++;
                    }
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    std::vector<float> rewriteQueryByChampionLayer(
        const void* query_data,
        int query_category) 
    {
        const float* q = (const float*)query_data;
        const size_t dim = getVectorDim();

        std::vector<float> rewritten(dim, 0.0f);

        if (!enable_query_rewrite_) {
            std::memcpy(rewritten.data(), q, dim * sizeof(float));
            return rewritten;
        }

        auto ep_it = champion_layer_entrypoints_.find(query_category);
        if (ep_it == champion_layer_entrypoints_.end()) {
            std::memcpy(rewritten.data(), q, dim * sizeof(float));
            return rewritten;
        }

        tableint entry_id = ep_it->second;
        if (isMarkedDeleted(entry_id) || element_levels_[entry_id] < champion_layer_level_) {
            std::memcpy(rewritten.data(), q, dim * sizeof(float));
            return rewritten;
        }

        // 本来就在同类型中，所以不做热启动和多跳邻居搜索
        auto top_candidates = searchBaseLayer_WithCategory(
            entry_id,
            query_data,
            champion_layer_level_,
            query_category,
            champion_rewrite_topk_,
            1,   // 这里对非0层不会生效，但参数保留即可
            0
        );

        if (top_candidates.empty()) {
            std::memcpy(rewritten.data(), q, dim * sizeof(float));
            return rewritten;
        }

        std::vector<tableint> selected_ids;
        while (!top_candidates.empty()) {
            selected_ids.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        // 逐维平均，注意一定按“实际找到的数量”平均
        for (tableint cid : selected_ids) {
            const float* v = (const float*)getDataByInternalId(cid);
            for (size_t i = 0; i < dim; ++i) {
                rewritten[i] += v[i];
            }
        }

        const float inv = 1.0f / (float)selected_ids.size();
        for (size_t i = 0; i < dim; ++i) {
            rewritten[i] *= inv;
        }

        return rewritten;
    }

    std::vector<float> rewriteQueryByORGN(
        const void* query_data,
        int query_category)
    {
        const float* q = static_cast<const float*>(query_data);
        const size_t dim = getVectorDim();
        const size_t element_count = cur_element_count.load();

        std::vector<float> rewritten(dim, 0.0f);

        // 出现异常情况时返回原始查询向量。
        auto return_original_query = [&]() -> std::vector<float> {
            std::memcpy(
                rewritten.data(),
                q,
                dim * sizeof(float)
            );
            return rewritten;
        };

        if (!enable_query_rewrite_ ||
            element_count == 0) {
            return return_original_query();
        }

        /*
        * 并查集用于计算合格诱导子图中每个节点所在连通分量的大小。
        *
        * HNSW 的邻接表在实现层面不一定严格对称，因此只要底层存在
        * u -> v 或 v -> u，就将二者视为连通。这对应无向诱导子图中的
        * 弱连通分量。
        */
        struct DisjointSet {
            std::vector<tableint> parent;
            std::vector<size_t> component_size;

            explicit DisjointSet(size_t count)
                : parent(
                    count,
                    std::numeric_limits<tableint>::max()),
                component_size(count, 0) {
            }

            void activate(tableint id) {
                parent[id] = id;
                component_size[id] = 1;
            }

            bool isActive(tableint id) const {
                return parent[id] !=
                    std::numeric_limits<tableint>::max();
            }

            tableint find(tableint id) {
                tableint root = id;

                while (parent[root] != root) {
                    root = parent[root];
                }

                // 路径压缩。
                while (parent[id] != id) {
                    const tableint next = parent[id];
                    parent[id] = root;
                    id = next;
                }

                return root;
            }

            void unite(tableint left, tableint right) {
                tableint left_root = find(left);
                tableint right_root = find(right);

                if (left_root == right_root) {
                    return;
                }

                // 按连通分量大小合并。
                if (component_size[left_root] <
                    component_size[right_root]) {
                    std::swap(left_root, right_root);
                }

                parent[right_root] = left_root;
                component_size[left_root] +=
                    component_size[right_root];
            }

            size_t getComponentSize(tableint id) {
                return component_size[find(id)];
            }
        };

        DisjointSet disjoint_set(element_count);
        std::vector<tableint> qualified_ids;

        /*
        * 第一步：扫描整个数据集，得到合格节点集合 X_Q。
        *
        * 在当前二元类别实现中：
        * getCategoryByInternalId(id) == query_category
        * 就等价于节点满足查询结构条件。
        */
        for (size_t raw_id = 0;
            raw_id < element_count;
            ++raw_id) {

            const tableint id =
                static_cast<tableint>(raw_id);

            if (isMarkedDeleted(id)) {
                continue;
            }

            if (getCategoryByInternalId(id) !=
                query_category) {
                continue;
            }

            disjoint_set.activate(id);
            qualified_ids.push_back(id);
        }

        if (qualified_ids.empty()) {
            return return_original_query();
        }

        /*
        * 第二步：扫描 HNSW 第 0 层的边，构造合格诱导子图 G_Q。
        *
        * 仅当两个端点均属于 X_Q 时才保留该边。
        * ORGN 对应完整的底层图，所以这里不能使用冠军层。
        */
        for (tableint id : qualified_ids) {
            linklistsizeint* link_list =
                get_linklist0(id);

            const size_t neighbor_count =
                getListCount(link_list);

            tableint* neighbors =
                reinterpret_cast<tableint*>(
                    link_list + 1
                );

            for (size_t j = 0;
                j < neighbor_count;
                ++j) {

                const tableint neighbor_id =
                    neighbors[j];

                if (neighbor_id >= element_count) {
                    continue;
                }

                if (!disjoint_set.isActive(neighbor_id)) {
                    continue;
                }

                disjoint_set.unite(
                    id,
                    neighbor_id
                );
            }
        }

        struct ORGNCandidate {
            tableint id;
            dist_t distance;
            size_t connectivity;
            size_t distance_rank;
            size_t connectivity_rank;
        };

        std::vector<ORGNCandidate> candidates;
        candidates.reserve(qualified_ids.size());

        /*
        * 第三步：计算所有合格节点与原始查询向量之间的距离，
        * 同时取得节点所在连通分量的大小。
        */
        for (tableint id : qualified_ids) {
            const dist_t distance = fstdistfunc_(
                query_data,
                getDataByInternalId(id),
                dist_func_param_
            );

            // 与原有统计口径保持一致。
            search_distance_computations_++;
            champion_rewrite_distance_computations_++;

            ORGNCandidate candidate;
            candidate.id = id;
            candidate.distance = distance;
            candidate.connectivity =
                disjoint_set.getComponentSize(id);
            candidate.distance_rank = 0;
            candidate.connectivity_rank = 0;

            candidates.push_back(candidate);
        }

        /*
        * 第四步：按查询距离升序排列。
        */
        std::vector<size_t> distance_order(
            candidates.size()
        );

        for (size_t i = 0;
            i < distance_order.size();
            ++i) {
            distance_order[i] = i;
        }

        std::sort(
            distance_order.begin(),
            distance_order.end(),
            [&](size_t left_index,
                size_t right_index) {

                const ORGNCandidate& left =
                    candidates[left_index];

                const ORGNCandidate& right =
                    candidates[right_index];

                if (left.distance != right.distance) {
                    return left.distance < right.distance;
                }

                // 保证相同距离下结果可复现。
                return left.id < right.id;
            }
        );

        /*
        * 使用竞争排名：
        * 例如 1, 2, 2, 4。
        * 距离相同的节点具有相同排名。
        */
        size_t current_distance_rank = 1;

        for (size_t position = 0;
            position < distance_order.size();
            ++position) {

            if (position > 0) {
                const dist_t previous_distance =
                    candidates[
                        distance_order[position - 1]
                    ].distance;

                const dist_t current_distance =
                    candidates[
                        distance_order[position]
                    ].distance;

                if (current_distance !=
                    previous_distance) {
                    current_distance_rank =
                        position + 1;
                }
            }

            candidates[
                distance_order[position]
            ].distance_rank =
                current_distance_rank;
        }

        /*
        * 第五步：按连通分量大小降序排列。
        */
        std::vector<size_t> connectivity_order(
            candidates.size()
        );

        for (size_t i = 0;
            i < connectivity_order.size();
            ++i) {
            connectivity_order[i] = i;
        }

        std::sort(
            connectivity_order.begin(),
            connectivity_order.end(),
            [&](size_t left_index,
                size_t right_index) {

                const ORGNCandidate& left =
                    candidates[left_index];

                const ORGNCandidate& right =
                    candidates[right_index];

                if (left.connectivity !=
                    right.connectivity) {
                    return left.connectivity >
                        right.connectivity;
                }

                return left.id < right.id;
            }
        );

        /*
        * 连通性必须采用并列排名。
        *
        * 同一连通分量内的节点具有相同连通性，不能再根据 internal_id
        * 强行赋予不同排名，否则融合结果会无意义地依赖节点插入顺序。
        */
        size_t current_connectivity_rank = 1;

        for (size_t position = 0;
            position < connectivity_order.size();
            ++position) {

            if (position > 0) {
                const size_t previous_connectivity =
                    candidates[
                        connectivity_order[position - 1]
                    ].connectivity;

                const size_t current_connectivity =
                    candidates[
                        connectivity_order[position]
                    ].connectivity;

                if (current_connectivity !=
                    previous_connectivity) {
                    current_connectivity_rank =
                        position + 1;
                }
            }

            candidates[
                connectivity_order[position]
            ].connectivity_rank =
                current_connectivity_rank;
        }

        /*
        * 第六步：从距离排序中选择最近的 R 个节点。
        *
        * 这里沿用原冠军层函数中的 champion_rewrite_topk_，
        * 不增加新的函数输入参数。
        */
        // const size_t R = std::min(
        //     champion_rewrite_topk_,
        //     candidates.size()
        // );
        const size_t R = candidates.size();
        if (R == 0) {
            return return_original_query();
        }

        std::vector<double> weighted_sum(
            dim,
            0.0
        );

        double weight_sum = 0.0;

        /*
        * 使用论文第 3.3 节的排名乘积倒数作为未归一化权重：
        *
        * w_i = 1 /
        *       (r_dist(i) * r_conn(i)).
        */
        for (size_t position = 0;
            position < R;
            ++position) {

            const ORGNCandidate& candidate =
                candidates[
                    distance_order[position]
                ];

            const double raw_weight =
                1.0 /
                (
                    static_cast<double>(
                        candidate.distance_rank
                    ) *
                    static_cast<double>(
                        candidate.connectivity_rank
                    )
                );

            const float* vector =
                reinterpret_cast<const float*>(
                    getDataByInternalId(
                        candidate.id
                    )
                );

            for (size_t d = 0;
                d < dim;
                ++d) {
                weighted_sum[d] +=
                    raw_weight *
                    static_cast<double>(vector[d]);
            }

            weight_sum += raw_weight;
        }

        if (!(weight_sum > 0.0)) {
            return return_original_query();
        }

        // 对权重归一化，得到最终的查询改写向量 p_Q。
        for (size_t d = 0;
            d < dim;
            ++d) {
            rewritten[d] =
                static_cast<float>(
                    weighted_sum[d] /
                    weight_sum
                );
        }

        return rewritten;
    }    

    std::vector<float> rewriteQueryByRGN(
        const void* query_data,
        int query_category,
        int rgn_rewrite_hop_count)
    {
        const float* q = static_cast<const float*>(query_data);
        const size_t dim = getVectorDim();
        const size_t element_count = cur_element_count.load();

        std::vector<float> rewritten(dim, 0.0f);

        /*
        * 保持原函数的回退语义：
        * 当查询改写被关闭、入口不存在或没有找到合格候选时，
        * 直接返回原始查询向量。
        */
        auto return_original_query = [&]() -> std::vector<float> {
            std::memcpy(
                rewritten.data(),
                q,
                dim * sizeof(float)
            );
            return rewritten;
        };

        if (!enable_query_rewrite_ ||
            element_count == 0 ||
            champion_layer_level_ < 1) {
            return return_original_query();
        }

        if (rgn_rewrite_hop_count < 0) {
            throw std::invalid_argument(
                "rgn_rewrite_hop_count must be non-negative"
            );
        }

        /*
        * 第一步：从代表表中获得当前查询类别在冠军层的入口节点。
        *
        * 当前代码采用整数类别表示结构属性集合，因此精确类别匹配
        * 等价于论文中的 A(q) 包含关系测试。
        */
        const auto entrypoint_it =
            champion_layer_entrypoints_.find(query_category);

        if (entrypoint_it == champion_layer_entrypoints_.end()) {
            return return_original_query();
        }

        const tableint entry_id = entrypoint_it->second;

        if (entry_id >= element_count ||
            isMarkedDeleted(entry_id) ||
            element_levels_[entry_id] < champion_layer_level_) {
            return return_original_query();
        }

        /*
        * 第二步：从入口节点出发，在冠军层进行 h 跳扩展。
        *
        * 这里不能在扩展过程中进行类别过滤。论文中的 RGN 是先在共享
        * guidance graph G_s 中扩展，再对扩展结果进行属性过滤。
        *
        * 如果在扩展过程中只允许同类别节点继续传播，就可能因为冠军层
        * 中的异构节点而提前中断，无法获得完整的 h 跳局部区域。
        */
        VisitedList* visited_list =
            visited_list_pool_->getFreeVisitedList();

        vl_type* visited_array = visited_list->mass;
        const vl_type visited_array_tag = visited_list->curV;

        std::queue<std::pair<tableint, int>> expansion_queue;
        std::vector<tableint> local_nodes;

        expansion_queue.emplace(entry_id, 0);
        visited_array[entry_id] = visited_array_tag;

        while (!expansion_queue.empty()) {
            const tableint node_id = expansion_queue.front().first;
            const int current_hop = expansion_queue.front().second;
            expansion_queue.pop();

            /*
            * 已删除节点不属于当前 guidance graph。
            * 正常情况下，删除节点不会被加入队列；这里再检查一次，
            * 防止查询期间索引状态发生变化。
            */
            if (isMarkedDeleted(node_id) ||
                element_levels_[node_id] < champion_layer_level_) {
                continue;
            }

            local_nodes.push_back(node_id);

            if (current_hop >= rgn_rewrite_hop_count) {
                continue;
            }

            linklistsizeint* link_list =
                get_linklist(node_id, champion_layer_level_);

            const size_t neighbor_count =
                getListCount(link_list);

            tableint* neighbors =
                reinterpret_cast<tableint*>(link_list + 1);

            for (size_t j = 0; j < neighbor_count; ++j) {
                const tableint neighbor_id = neighbors[j];

                if (neighbor_id >= element_count ||
                    visited_array[neighbor_id] == visited_array_tag) {
                    continue;
                }

                visited_array[neighbor_id] = visited_array_tag;

                if (isMarkedDeleted(neighbor_id) ||
                    element_levels_[neighbor_id] <
                        champion_layer_level_) {
                    continue;
                }

                expansion_queue.emplace(
                    neighbor_id,
                    current_hop + 1
                );
            }
        }

        visited_list_pool_->releaseVisitedList(visited_list);

        if (local_nodes.empty()) {
            return return_original_query();
        }

        /*
        * 第三步：对 h 跳局部区域进行属性过滤，得到：
        *
        * X_Q^{loc}
        * =
        * {e_i in C | A(q) subseteq A(e_i)}。
        *
        * 当前实现中，复合结构属性已经编码为整数类别，因此类别相等
        * 就是当前代码对应的合格性判断。
        */
        std::vector<tableint> qualified_ids;
        qualified_ids.reserve(local_nodes.size());

        std::unordered_set<tableint> qualified_set;
        qualified_set.reserve(local_nodes.size() * 2 + 1);

        for (tableint node_id : local_nodes) {
            if (getCategoryByInternalId(node_id) !=
                query_category) {
                continue;
            }

            qualified_ids.push_back(node_id);
            qualified_set.insert(node_id);
        }

        if (qualified_ids.empty()) {
            return return_original_query();
        }

        struct RGNCandidate {
            tableint id;
            dist_t distance;
            size_t local_connectivity;
            size_t distance_rank;
            size_t connectivity_rank;
        };

        std::vector<RGNCandidate> candidates;
        candidates.reserve(qualified_ids.size());

        /*
        * 第四步：计算每个局部合格节点到原始查询向量 q 的距离。
        */
        for (tableint node_id : qualified_ids) {
            const dist_t distance = fstdistfunc_(
                query_data,
                getDataByInternalId(node_id),
                dist_func_param_
            );

            /*
            * 与 ORGN 的统计口径保持一致。
            */
            search_distance_computations_++;
            champion_rewrite_distance_computations_++;

            RGNCandidate candidate;
            candidate.id = node_id;
            candidate.distance = distance;
            candidate.local_connectivity = 0;
            candidate.distance_rank = 0;
            candidate.connectivity_rank = 0;

            candidates.push_back(candidate);
        }

        /*
        * 第五步：构造局部合格诱导子图 G_Q^{loc}，并计算：
        *
        * c_Q^{loc}(e_i)
        * =
        * |N_{G_Q^{loc}}(e_i)|。
        *
        * 即只统计：
        * 1. 位于当前 h 跳局部区域；
        * 2. 满足查询结构属性；
        * 3. 与当前节点在冠军层直接相邻
        * 的节点数量。
        */
        for (RGNCandidate& candidate : candidates) {
            linklistsizeint* link_list =
                get_linklist(
                    candidate.id,
                    champion_layer_level_
                );

            const size_t neighbor_count =
                getListCount(link_list);

            tableint* neighbors =
                reinterpret_cast<tableint*>(link_list + 1);

            size_t qualified_neighbor_count = 0;

            for (size_t j = 0; j < neighbor_count; ++j) {
                const tableint neighbor_id = neighbors[j];

                if (neighbor_id == candidate.id) {
                    continue;
                }

                if (qualified_set.find(neighbor_id) !=
                    qualified_set.end()) {
                    ++qualified_neighbor_count;
                }
            }

            candidate.local_connectivity =
                qualified_neighbor_count;
        }

        /*
        * 第六步：按照查询距离升序排列。
        */
        std::vector<size_t> distance_order(
            candidates.size()
        );

        for (size_t i = 0;
            i < distance_order.size();
            ++i) {
            distance_order[i] = i;
        }

        std::sort(
            distance_order.begin(),
            distance_order.end(),
            [&](size_t left_index, size_t right_index) {
                const RGNCandidate& left =
                    candidates[left_index];

                const RGNCandidate& right =
                    candidates[right_index];

                if (left.distance < right.distance) {
                    return true;
                }

                if (right.distance < left.distance) {
                    return false;
                }

                /*
                * 相同距离下使用内部 ID 保证结果可复现。
                * 内部 ID 不参与排名数值的计算。
                */
                return left.id < right.id;
            }
        );

        /*
        * 使用 competition ranking。
        *
        * 例如距离序列中的排名为：
        * 1, 2, 2, 4。
        *
        * 距离相同的节点获得相同排名。
        */
        size_t current_distance_rank = 1;

        for (size_t position = 0;
            position < distance_order.size();
            ++position) {
            if (position > 0) {
                const dist_t previous_distance =
                    candidates[
                        distance_order[position - 1]
                    ].distance;

                const dist_t current_distance =
                    candidates[
                        distance_order[position]
                    ].distance;

                if (current_distance != previous_distance) {
                    current_distance_rank =
                        position + 1;
                }
            }

            candidates[
                distance_order[position]
            ].distance_rank = current_distance_rank;
        }

        /*
        * 第七步：按照局部连通性降序排列。
        *
        * 局部合格邻居越多，说明该节点所在的合格区域越适合作为
        * 第一阶段导航的目标区域。
        */
        std::vector<size_t> connectivity_order(
            candidates.size()
        );

        for (size_t i = 0;
            i < connectivity_order.size();
            ++i) {
            connectivity_order[i] = i;
        }

        std::sort(
            connectivity_order.begin(),
            connectivity_order.end(),
            [&](size_t left_index, size_t right_index) {
                const RGNCandidate& left =
                    candidates[left_index];

                const RGNCandidate& right =
                    candidates[right_index];

                if (left.local_connectivity !=
                    right.local_connectivity) {
                    return left.local_connectivity >
                        right.local_connectivity;
                }

                return left.id < right.id;
            }
        );

        /*
        * 连通性相同的节点必须获得相同排名。
        *
        * 不能使用内部 ID 打破连通性排名，否则最终融合权重会依赖
        * 数据的插入顺序。
        */
        size_t current_connectivity_rank = 1;

        for (size_t position = 0;
            position < connectivity_order.size();
            ++position) {
            if (position > 0) {
                const size_t previous_connectivity =
                    candidates[
                        connectivity_order[position - 1]
                    ].local_connectivity;

                const size_t current_connectivity =
                    candidates[
                        connectivity_order[position]
                    ].local_connectivity;

                if (current_connectivity !=
                    previous_connectivity) {
                    current_connectivity_rank =
                        position + 1;
                }
            }

            candidates[
                connectivity_order[position]
            ].connectivity_rank =
                current_connectivity_rank;
        }

        /*
        * 第八步：按照论文中的排名加权公式进行融合。
        *
        * raw_weight_i
        * =
        * 1 /
        * (r_dist(i) * r_conn(i))
        *
        * alpha_i
        * =
        * raw_weight_i /
        * sum_j(raw_weight_j)
        *
        * p_Q
        * =
        * sum_i(alpha_i * x_i)
        *
        * 严格按照论文公式，这里融合 X_Q^{loc} 中的全部节点。
        */
        std::vector<double> weighted_sum(
            dim,
            0.0
        );

        double total_weight = 0.0;

        for (const RGNCandidate& candidate :
            candidates) {
            const double distance_rank =
                static_cast<double>(
                    candidate.distance_rank
                );

            const double connectivity_rank =
                static_cast<double>(
                    candidate.connectivity_rank
                );

            const double raw_weight =
                1.0 /
                (
                    distance_rank *
                    connectivity_rank
                );

            const float* candidate_vector =
                reinterpret_cast<const float*>(
                    getDataByInternalId(candidate.id)
                );

            for (size_t d = 0; d < dim; ++d) {
                weighted_sum[d] +=
                    raw_weight *
                    static_cast<double>(
                        candidate_vector[d]
                    );
            }

            total_weight += raw_weight;
        }

        if (!(total_weight > 0.0)) {
            return return_original_query();
        }

        for (size_t d = 0; d < dim; ++d) {
            rewritten[d] =
                static_cast<float>(
                    weighted_sum[d] /
                    total_weight
                );
        }

        return rewritten;
    }

    // ===================== 4. searchKnn 搜索用带类别过滤（功能正常） =====================
    std::vector<std::pair<labeltype, dist_t>> searchKnn(const void *query_data, size_t k, int query_category, int hop_count = 1, int init_hop_count = 0,int query_rewrite_method = 0,int rgn_rewrite_hop_count = 2) {
        // std::vector<float> rewritten_query = rewriteQueryByChampionLayer(query_data, query_category);
        // std::vector<float> rewritten_query = rewriteQueryByORGN(query_data, query_category);
        // std::vector<float> rewritten_query = rewriteQueryByRGN(query_data, query_category);
        std::vector<float> rewritten_query;

        if (query_rewrite_method == 0) {
            // 原始方法：冠军层候选节点简单平均
            rewritten_query =
                rewriteQueryByChampionLayer(query_data, query_category);

        } else if (query_rewrite_method == 1) {
            // ORGN：全局候选节点排序加权融合
            rewritten_query =
                rewriteQueryByORGN(query_data, query_category);

        } else if (query_rewrite_method == 2) {
            // RGN：冠军层局部多跳邻居排序加权融合
            rewritten_query =
                rewriteQueryByRGN(
                    query_data,
                    query_category,
                    rgn_rewrite_hop_count
                );

        } else {
            throw std::invalid_argument(
                "query_rewrite_method must be 0, 1, or 2"
            );
        }        
        const void* high_level_query = (const void*)rewritten_query.data();
        const void* low_level_query = query_data;
        std::vector<std::pair<labeltype, dist_t>> result;
        if (cur_element_count == 0 || k == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist;

        if (isMarkedDeleted(currObj)) {
            currObj = -1;
            curdist = std::numeric_limits<dist_t>::max();
        } else {
            curdist = fstdistfunc_(high_level_query, getDataByInternalId(currObj), dist_func_param_);
            search_distance_computations_++;
        }

        // 高层导航：原生逻辑
        if ((signed)currObj != -1) {
            bool current_matches_query_category =
                !isMarkedDeleted(currObj) &&
                getCategoryByInternalId(currObj) == query_category;
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;

                    tableint *datal = (tableint *) (data + 1);
                    size_t qualified_neighbor_count = 0;
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand >= max_elements_)
                            throw std::runtime_error("cand error");
                        
                        if (isMarkedDeleted(cand)) {
                            continue;
                        }

                        if (acorn_build_mode_ == 1) {
                            // ACORN-gamma scans the dense M*gamma list,
                            // filters by the structured predicate, and only
                            // traverses the first M qualifying neighbors.
                            if (getCategoryByInternalId(cand) !=
                                query_category) {
                                continue;
                            }
                            ++qualified_neighbor_count;
                        }

                        dist_t d = fstdistfunc_(high_level_query, getDataByInternalId(cand), dist_func_param_);
                        metric_distance_computations++;
                        search_distance_computations_++;  // 新增：统计搜索距离计算次数
                        if ((acorn_build_mode_ == 1 &&
                             !current_matches_query_category) ||
                            d < curdist) {
                            curdist = d;
                            currObj = cand;
                            current_matches_query_category = true;
                            changed = true;
                        }

                        if (acorn_build_mode_ == 1 &&
                            qualified_neighbor_count >= M_) {
                            break;
                        }
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        // 🔴 底层用带类别过滤的4参数函数
        if ((signed)currObj != -1) {
            top_candidates = searchBaseLayer_WithCategory(
                currObj,
                low_level_query,
                0,
                query_category,
                std::max(ef_, k),
                hop_count,
                init_hop_count
            );
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (!top_candidates.empty()) {
            auto rez = top_candidates.top();
            result.emplace_back(getExternalLabel(rez.second), rez.first);
            top_candidates.pop();
        }

        return result;
    }

    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }
};
}  // namespace hnswlib
