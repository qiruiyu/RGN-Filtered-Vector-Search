    std::vector<std::pair<labeltype, dist_t>> searchKnn(
        const void *query_data,
        size_t k,
        const AttributeSet& query_attributes,
        int hop_count = 1,
        int query_rewrite_method = 0,
        int rgn_rewrite_hop_count = 2,
        int rgn_fusion_strategy = 0,
        int two_phase_query_mode = 1) {
        
        
        
        if (two_phase_query_mode < 0 || two_phase_query_mode > 3) {
            throw std::invalid_argument(
                "two_phase_query_mode must be 0, 1, 2, or 3"
            );
        }

        
        
        
        const bool use_rewritten_query_at_high_level =
            (two_phase_query_mode & 1) != 0;
        const bool use_rewritten_query_at_low_level =
            (two_phase_query_mode & 2) != 0;
        std::vector<float> rewritten_query;

        if (use_rewritten_query_at_high_level ||
            use_rewritten_query_at_low_level) {
            if (query_rewrite_method == 0) {
                
                rewritten_query =
                    rewriteQueryByGuidanceLayer(query_data, query_attributes);

            } else if (query_rewrite_method == 1) {
                
                rewritten_query =
                    rewriteQueryByORGN(query_data, query_attributes);

            } else if (query_rewrite_method == 2) {
                
                rewritten_query =
                    rewriteQueryByRGN(
                        query_data,
                        query_attributes,
                        rgn_rewrite_hop_count,
                        rgn_fusion_strategy
                    );

            } else {
                throw std::invalid_argument(
                    "query_rewrite_method must be 0, 1, or 2"
                );
            }
        }

        const void* rewritten_query_data =
            static_cast<const void*>(rewritten_query.data());
        const void* high_level_query = use_rewritten_query_at_high_level
            ? rewritten_query_data
            : query_data;
        const void* low_level_query = use_rewritten_query_at_low_level
            ? rewritten_query_data
            : query_data;
        std::vector<std::pair<labeltype, dist_t>> result;
        if (cur_element_count == 0 || k == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(
            high_level_query,
            getDataByInternalId(currObj),
            dist_func_param_);
        search_distance_computations_++;

        
        if ((signed)currObj != -1) {
            bool current_matches_query_attributes =
                satisfies(
                    getAttributesByInternalId(currObj),
                    query_attributes);
            for (int level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    size_t qualified_neighbor_count = 0;
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand >= max_elements_)
                            throw std::runtime_error("cand error");
                        if (acorn_build_mode_ == 1) {
                            
                            
                            
                            if (!satisfies(
                                    getAttributesByInternalId(cand),
                                    query_attributes)) {
                                continue;
                            }
                            ++qualified_neighbor_count;
                        }

                        dist_t d = fstdistfunc_(high_level_query, getDataByInternalId(cand), dist_func_param_);
                        search_distance_computations_++;  
                        if ((acorn_build_mode_ == 1 &&
                             !current_matches_query_attributes) ||
                            d < curdist) {
                            curdist = d;
                            currObj = cand;
                            current_matches_query_attributes = true;
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
        
        if ((signed)currObj != -1) {
            top_candidates = searchBaseLayerWithAttributes(
                currObj,
                low_level_query,
                0,
                query_attributes,
                std::max(ef_, k),
                hop_count
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

