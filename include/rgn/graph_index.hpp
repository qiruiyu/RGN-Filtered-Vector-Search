// Derived from nmslib/hnswlib v0.8.0 and modified for RGN.
// Licensed under Apache-2.0; see LICENSE and THIRD_PARTY_NOTICES.md.
#pragma once

#include "attributes.hpp"
#include "region_guidance.hpp"
#include "visited_list_pool.hpp"
#include "core.hpp"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <memory>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <limits>
#include <cstring>
#include <cmath>
#include <fstream>

namespace rgn {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template<typename dist_t>
class RgnIndex {
 public:
    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    
    int acorn_build_mode_{0};
    size_t acorn_gamma_{1};
    size_t acorn_m_beta_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    
    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  
    std::vector<AttributeSet> element_attributes_;

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine representative_probability_generator_;

    mutable std::atomic<long> search_distance_computations_{0};

    
    mutable std::atomic<uint64_t> build_distance_computations_{0};

    
    mutable std::atomic<uint16_t> search_top_candidates_count_{0};

    
    
    bool enable_query_rewrite_{true};

    
    int guidance_layer_level_{1};

    
    size_t guidance_layer_min_per_attribute_set_{10};

    
    
    double guidance_layer_ratio_{-1.0};

    
    uint32_t guidance_layer_assignment_version_{2};

    
    size_t guidance_rewrite_topk_{10};

    mutable std::map<AttributeSet, tableint> representative_table_;

    
    
    
    
    uint32_t representative_selection_strategy_{0};
    static constexpr double representative_sampling_probability_ = 0.3;

    struct RepresentativeMeanState {
        uint64_t participant_count{0};
        std::vector<float> mean;
    };

    
    std::map<AttributeSet, RepresentativeMeanState>
        representative_mean_states_;
    mutable std::mutex representative_table_lock_;

    
    
    mutable std::map<AttributeSet, size_t> representative_forced_count_;

    
    mutable std::atomic<uint64_t> guidance_rewrite_distance_computations_{0};

    
    mutable std::atomic<uint64_t> guidance_layer_build_distance_computations_{0};


    RgnIndex(SpaceInterface<dist_t> *s) {
    }


    RgnIndex(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool = false,
        size_t max_elements = 0) {
        loadIndex(location, s, max_elements);
    }


    RgnIndex(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        int acorn_build_mode = 0,
        size_t acorn_gamma = 1,
        size_t acorn_m_beta = 0)
        : link_list_locks_(max_elements),
            element_levels_(max_elements),
            element_attributes_(max_elements) {
        max_elements_ = max_elements;
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
        
        
        
        
        maxM_ = expanded_M;
        maxM0_ = 2 * expanded_M;

        
        
        const size_t construction_candidate_budget = expanded_M;
        ef_construction_ = std::max(
            ef_construction,
            construction_candidate_budget);
        ef_ = 10;
        
        level_generator_.seed(random_seed);
        representative_probability_generator_.seed(random_seed + 2);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: RgnIndex failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~RgnIndex() {
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

     
    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        
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

    void setGuidanceLayerLevel(int lc) {
        if (lc < 1) {
            throw std::runtime_error("guidance layer level must be >= 1");
        }
        guidance_layer_level_ = lc;
    }

    int getGuidanceLayerLevel() const {
        return guidance_layer_level_;
    }

    void setGuidanceLayerMinPerAttributeSet(size_t minimum) {
        guidance_layer_min_per_attribute_set_ = minimum;
    }

    size_t getGuidanceLayerMinPerAttributeSet() const {
        return guidance_layer_min_per_attribute_set_;
    }

    void setGuidanceLayerRatio(double ratio) {
        if (!std::isfinite(ratio) ||
            (ratio != -1.0 && (ratio < 0.0 || ratio > 1.0))) {
            throw std::invalid_argument(
                "guidance layer ratio must be -1 or in [0, 1]");
        }
        guidance_layer_ratio_ = ratio;
    }

    double getGuidanceLayerRatio() const {
        return guidance_layer_ratio_;
    }

    uint32_t getGuidanceLayerAssignmentVersion() const {
        return guidance_layer_assignment_version_;
    }

    void setRepresentativeSelectionStrategy(uint32_t strategy) {
        if (strategy > 2) {
            throw std::invalid_argument(
                "guidance layer entrypoint selection strategy must be 0, 1, or 2");
        }
        if (cur_element_count.load() != 0 &&
            strategy != representative_selection_strategy_) {
            throw std::runtime_error(
                "cannot change guidance layer entrypoint selection strategy after insertion starts");
        }
        if (strategy != representative_selection_strategy_) {
            representative_mean_states_.clear();
        }
        representative_selection_strategy_ = strategy;
    }

    uint32_t getRepresentativeSelectionStrategy() const {
        return representative_selection_strategy_;
    }

    void setGuidanceRewriteTopk(size_t topk) {
        guidance_rewrite_topk_ = topk;
    }

    size_t getGuidanceRewriteTopk() const {
        return guidance_rewrite_topk_;
    }

    bool hasGuidanceEntrypoint(const AttributeSet& query_attributes) const {
        for (const auto& entry : representative_table_) {
            if (satisfies(entry.first, query_attributes)) {
                return true;
            }
        }
        return false;
    }

    tableint getGuidanceEntrypoint(const AttributeSet& query_attributes) const {
        for (const auto& entry : representative_table_) {
            if (satisfies(entry.first, query_attributes)) {
                return entry.second;
            }
        }
        throw std::runtime_error("No guidance entrypoint satisfies the query attributes");
    }

    uint64_t getGuidanceRewriteDistanceComputations() const {
        return guidance_rewrite_distance_computations_.load();
    }

    void resetGuidanceRewriteDistanceComputations() {
        guidance_rewrite_distance_computations_.store(0);
    }

    inline const AttributeSet& getAttributesByInternalId(tableint internal_id) const {
        if (internal_id >= element_attributes_.size()) {
            throw std::out_of_range("internal id exceeds attribute storage");
        }
        return element_attributes_[internal_id];
    }

    bool matchesStoredEntity(
        labeltype label,
        const void* data,
        const AttributeSet& attributes) const {
        const auto found = label_lookup_.find(label);
        if (found == label_lookup_.end()) {
            return false;
        }
        const tableint internal_id = found->second;
        return getAttributesByInternalId(internal_id) == attributes &&
            std::memcmp(getDataByInternalId(internal_id), data, data_size_) == 0;
    }

    inline void setAttributesByInternalId(
        tableint internal_id,
        const AttributeSet& attributes) {
        if (internal_id >= element_attributes_.size()) {
            throw std::out_of_range("internal id exceeds attribute storage");
        }
        element_attributes_[internal_id] = attributes;
    }

    
    long getSearchDistanceComputations() const {
        return search_distance_computations_.load(); 
    }

    
    void resetSearchDistanceComputations() {
        search_distance_computations_.store(0); 
    }

    
    uint64_t getBuildDistanceComputations() const {
        return build_distance_computations_.load(); 
    }

    
    void resetBuildDistanceComputations() {
        build_distance_computations_.store(0); 
    }

    uint64_t getGuidanceLayerBuildDistanceComputations() const {
        return guidance_layer_build_distance_computations_.load();
    }

    void resetGuidanceLayerBuildDistanceComputations() {
        guidance_layer_build_distance_computations_.store(0);
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

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound = fstdistfunc_(
            data_point,
            getDataByInternalId(ep_id),
            dist_func_param_);
        build_distance_computations_++;
        if (layer == guidance_layer_level_) {
            guidance_layer_build_distance_computations_++;
        }
        top_candidates.emplace(lowerBound, ep_id);
        candidateSet.emplace(-lowerBound, ep_id);
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

            int *data;  
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);

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

#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                build_distance_computations_++;   
                if (layer == guidance_layer_level_) {
                    guidance_layer_build_distance_computations_++;
                }
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

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

    
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerWithAttributes(
        tableint ep_id,
        const void *data_point,
        int layer,
        const AttributeSet& query_attributes,
        size_t ef,
        int hop_count
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

        candidate_set.emplace(-std::numeric_limits<dist_t>::max(), ep_id);
        visited_array[ep_id] = visited_array_tag;

        if (layer == 0) {
            std::queue<std::pair<tableint, int>> seed_queue;
            seed_queue.emplace(ep_id, 0);

            while (!seed_queue.empty()) {
                const auto [node_id, depth] = seed_queue.front();
                seed_queue.pop();

                if (satisfies(
                        getAttributesByInternalId(node_id),
                        query_attributes)) {
                    const dist_t distance = fstdistfunc_(
                        data_point,
                        getDataByInternalId(node_id),
                        dist_func_param_);
                    search_distance_computations_++;
                    try_add_candidate(node_id, distance);
                }

                if (depth >= 1) {
                    continue;
                }

                int* seed_neighbors = (int*)get_linklist0(node_id);
                const size_t seed_neighbor_count =
                    getListCount((linklistsizeint*)seed_neighbors);
                for (size_t index = 1;
                     index <= seed_neighbor_count;
                     ++index) {
                    const tableint neighbor_id = *(seed_neighbors + index);
                    if (neighbor_id >= max_elements_ ||
                        visited_array[neighbor_id] == visited_array_tag) {
                        continue;
                    }
                    visited_array[neighbor_id] = visited_array_tag;
                    seed_queue.emplace(neighbor_id, depth + 1);
                }
            }
        } else if (satisfies(
                       getAttributesByInternalId(ep_id),
                       query_attributes)) {
            const dist_t distance = fstdistfunc_(
                data_point,
                getDataByInternalId(ep_id),
                dist_func_param_);
            search_distance_computations_++;
            guidance_rewrite_distance_computations_++;
            try_add_candidate(ep_id, distance);
        }

        const size_t TOP_KEEP =
            acorn_build_mode_ == 1 ? 2 * M_ : maxM0_;
        std::vector<std::pair<dist_t, tableint>> temp_neighbors;

        
        
        
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

                    if (satisfies(getAttributesByInternalId(nid), query_attributes)) {
                        const dist_t d = fstdistfunc_(
                            data_point,
                            getDataByInternalId(nid),
                            dist_func_param_);
                        search_distance_computations_++;
                        guidance_rewrite_distance_computations_++;
                        try_add_candidate(nid, d);
                    }
                }
                continue;
            }

            
            
            
            
            
            temp_neighbors.clear();
            std::queue<std::pair<tableint, int>> bfs_queue;

            
            
            
            
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

                if (satisfies(getAttributesByInternalId(node_id), query_attributes)) {
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

    
    
    void getNeighborsByHeuristic2(
        std::priority_queue<
            std::pair<dist_t, tableint>,
            std::vector<std::pair<dist_t, tableint>>,
            CompareByFirst>& top_candidates,
        size_t neighbor_limit,
        int construction_layer) {
        if (top_candidates.size() <= neighbor_limit) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> closest;
        while (!top_candidates.empty()) {
            closest.emplace(
                -top_candidates.top().first,
                top_candidates.top().second);
            top_candidates.pop();
        }

        std::vector<std::pair<dist_t, tableint>> selected;
        selected.reserve(neighbor_limit);
        while (!closest.empty() && selected.size() < neighbor_limit) {
            const auto candidate = closest.top();
            closest.pop();
            const dist_t distance_to_base = -candidate.first;
            bool keep = true;
            for (const auto& neighbor : selected) {
                const dist_t inter_neighbor_distance = fstdistfunc_(
                    getDataByInternalId(neighbor.second),
                    getDataByInternalId(candidate.second),
                    dist_func_param_);
                build_distance_computations_++;
                if (construction_layer == guidance_layer_level_) {
                    guidance_layer_build_distance_computations_++;
                }
                if (inter_neighbor_distance < distance_to_base) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                selected.push_back(candidate);
            }
        }

        for (const auto& neighbor : selected) {
            top_candidates.emplace(-neighbor.first, neighbor.second);
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
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        const size_t selected_neighbor_limit =
            acorn_build_mode_ == 1 ? maxM_ : M_;

        if (acorn_build_mode_ == 1) {
            getNearestNeighborsOrdered(
                top_candidates,
                selected_neighbor_limit);
        } else {
            getNeighborsByHeuristic2(
                top_candidates,
                M_,
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
            
            
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx])
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

            {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    build_distance_computations_++;  
                    if (level == guidance_layer_level_) {
                        guidance_layer_build_distance_computations_++;
                    }
                    
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                        build_distance_computations_++;  
                        if (level == guidance_layer_level_) {
                            guidance_layer_build_distance_computations_++;
                        }
                    }

                    if (acorn_build_mode_ == 1) {
                        getNearestNeighborsOrdered(
                            candidates,
                            Mcurmax);
                    } else {
                        getNeighborsByHeuristic2(
                            candidates,
                            Mcurmax,
                            level);
                    }
                    
                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    
                     










                }
            }
        }

        return next_closest_entry_point;
    }


    #include "rgn/index_persistence.tpp"
    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    #include "rgn/representative_table.tpp"
    tableint addPoint(
        const void *data_point,
        labeltype label,
        const AttributeSet& attributes,
        int level = -1) {

        tableint cur_c = 0;
        {
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                throw std::runtime_error("Duplicate labels are not supported");
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = decideLevelForNewPoint(attributes, level);

        registerRepresentative(cur_c, attributes, curlevel, data_point);

        
        element_levels_[cur_c] = curlevel;  

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);
        setAttributesByInternalId(cur_c, attributes);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                build_distance_computations_++;  
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
                            build_distance_computations_++;  
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                
                auto top_candidates = searchBaseLayer(currObj, data_point, level);

                currObj = mutuallyConnectNewElement(cur_c, top_candidates, level);
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

    #include "rgn/query_rewrite_mean.tpp"
    #include "rgn/orgn_navigation.tpp"
    #include "rgn/rgn_navigation.tpp"
    #include "rgn/hybrid_search.tpp"

};
}
