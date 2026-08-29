    std::vector<float> rewriteQueryByGuidanceLayer(
        const void* query_data,
        const AttributeSet& query_attributes)
    {
        const float* q = (const float*)query_data;
        const size_t dim = getVectorDim();

        std::vector<float> rewritten(dim, 0.0f);

        if (!enable_query_rewrite_) {
            std::memcpy(rewritten.data(), q, dim * sizeof(float));
            return rewritten;
        }

        auto ep_it = representative_table_.end();
        for (auto it = representative_table_.begin();
             it != representative_table_.end();
             ++it) {
            if (satisfies(it->first, query_attributes)) {
                ep_it = it;
                break;
            }
        }
        if (ep_it == representative_table_.end()) {
            std::memcpy(rewritten.data(), q, dim * sizeof(float));
            return rewritten;
        }

        tableint entry_id = ep_it->second;
        if (element_levels_[entry_id] < guidance_layer_level_) {
            std::memcpy(rewritten.data(), q, dim * sizeof(float));
            return rewritten;
        }

        
        auto top_candidates = searchBaseLayerWithAttributes(
            entry_id,
            query_data,
            guidance_layer_level_,
            query_attributes,
            guidance_rewrite_topk_,
            1
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

