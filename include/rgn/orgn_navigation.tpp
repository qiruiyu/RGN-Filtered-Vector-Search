    std::vector<float> rewriteQueryByORGN(
        const void* query_data,
        const AttributeSet& query_attributes)
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
            element_count == 0) {
            return return_original_query();
        }

         






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

         






        for (size_t raw_id = 0;
            raw_id < element_count;
            ++raw_id) {

            const tableint id =
                static_cast<tableint>(raw_id);

            if (!satisfies(
                    getAttributesByInternalId(id),
                    query_attributes)) {
                continue;
            }

            disjoint_set.activate(id);
            qualified_ids.push_back(id);
        }

        if (qualified_ids.empty()) {
            return return_original_query();
        }

         





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

         



        for (tableint id : qualified_ids) {
            const dist_t distance = fstdistfunc_(
                query_data,
                getDataByInternalId(id),
                dist_func_param_
            );

            
            search_distance_computations_++;
            guidance_rewrite_distance_computations_++;

            ORGNCandidate candidate;
            candidate.id = id;
            candidate.distance = distance;
            candidate.connectivity =
                disjoint_set.getComponentSize(id);
            candidate.distance_rank = 0;
            candidate.connectivity_rank = 0;

            candidates.push_back(candidate);
        }

         


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

         





        
        
        
        
        const size_t R = candidates.size();
        if (R == 0) {
            return return_original_query();
        }

        std::vector<double> weighted_sum(
            dim,
            0.0
        );

        double weight_sum = 0.0;

         





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

