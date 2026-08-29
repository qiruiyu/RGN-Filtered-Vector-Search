    std::vector<float> rewriteQueryByRGN(
        const void* query_data,
        const AttributeSet& query_attributes,
        int rgn_rewrite_hop_count,
        int rgn_fusion_strategy)
    {
        const float* q = static_cast<const float*>(query_data);
        const size_t dim = getVectorDim();
        const size_t element_count = cur_element_count.load();

        std::vector<float> rewritten(dim, 0.0f);

         




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
            guidance_layer_level_ < 1) {
            return return_original_query();
        }

        if (rgn_rewrite_hop_count < 0) {
            throw std::invalid_argument(
                "rgn_rewrite_hop_count must be non-negative"
            );
        }

        if (rgn_fusion_strategy < 0 ||
            rgn_fusion_strategy > 4) {
            throw std::invalid_argument(
                "rgn_fusion_strategy must be 0, 1, 2, 3, or 4"
            );
        }

         





        auto entrypoint_it = representative_table_.end();
        for (auto it = representative_table_.begin();
             it != representative_table_.end();
             ++it) {
            if (satisfies(it->first, query_attributes)) {
                entrypoint_it = it;
                break;
            }
        }

        if (entrypoint_it == representative_table_.end()) {
            return return_original_query();
        }

        const tableint entry_id = entrypoint_it->second;

        if (entry_id >= element_count ||
            element_levels_[entry_id] < guidance_layer_level_) {
            return return_original_query();
        }

         








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

             




            if (element_levels_[node_id] < guidance_layer_level_) {
                continue;
            }

            local_nodes.push_back(node_id);

            if (current_hop >= rgn_rewrite_hop_count) {
                continue;
            }

            linklistsizeint* link_list =
                get_linklist(node_id, guidance_layer_level_);

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

                if (element_levels_[neighbor_id] <
                        guidance_layer_level_) {
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

         









        std::vector<tableint> qualified_ids;
        qualified_ids.reserve(local_nodes.size());

        std::unordered_set<tableint> qualified_set;
        qualified_set.reserve(local_nodes.size() * 2 + 1);

        for (tableint node_id : local_nodes) {
            if (!satisfies(
                    getAttributesByInternalId(node_id),
                    query_attributes)) {
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

        const bool uses_distance_rank =
            rgn_fusion_strategy == 0 ||
            rgn_fusion_strategy == 1;

        const bool uses_connectivity_rank =
            rgn_fusion_strategy == 0 ||
            rgn_fusion_strategy == 2;

         




        for (tableint node_id : qualified_ids) {
            RGNCandidate candidate;
            candidate.id = node_id;
            candidate.distance = dist_t();
            candidate.local_connectivity = 0;
            candidate.distance_rank = 0;
            candidate.connectivity_rank = 0;

            if (uses_distance_rank) {
                candidate.distance = fstdistfunc_(
                    query_data,
                    getDataByInternalId(node_id),
                    dist_func_param_
                );

                 


                search_distance_computations_++;
                guidance_rewrite_distance_computations_++;
            }

            candidates.push_back(candidate);
        }

         












        if (uses_connectivity_rank) {
            for (RGNCandidate& candidate : candidates) {
                linklistsizeint* link_list =
                    get_linklist(
                        candidate.id,
                        guidance_layer_level_
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
        }

         


        if (uses_distance_rank) {
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

                     



                    return left.id < right.id;
                }
            );

             







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
        }

         





        if (uses_connectivity_rank) {
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
        }

         










        uint32_t random_seed = 2166136261u;
        const unsigned char* query_bytes =
            reinterpret_cast<const unsigned char*>(q);
        const size_t query_byte_count = dim * sizeof(float);
        for (size_t i = 0; i < query_byte_count; ++i) {
            random_seed ^= static_cast<uint32_t>(query_bytes[i]);
            random_seed *= 16777619u;
        }
        for (std::int32_t attribute : query_attributes.values()) {
            random_seed ^= static_cast<uint32_t>(attribute);
            random_seed *= 16777619u;
        }

        std::vector<GuidanceCandidate> fusion_candidates;
        fusion_candidates.reserve(candidates.size());
        for (const RGNCandidate& candidate : candidates) {
            fusion_candidates.push_back(GuidanceCandidate{
                candidate.id,
                static_cast<double>(candidate.distance),
                candidate.local_connectivity,
                candidate.distance_rank,
                candidate.connectivity_rank,
                reinterpret_cast<const float*>(getDataByInternalId(candidate.id))});
        }

        const FusionStrategy strategy =
            static_cast<FusionStrategy>(rgn_fusion_strategy);
        const size_t result_limit =
            strategy == FusionStrategy::Random ? 1 : fusion_candidates.size();
        rewritten = fuseTransitionVector(
            std::move(fusion_candidates),
            dim,
            result_limit,
            strategy,
            random_seed);
        if (rewritten.empty()) {
            return return_original_query();
        }
        return rewritten;
    }

    
