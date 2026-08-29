    int decideLevelForNewPoint(
        const AttributeSet& attributes,
        int user_level = -1) {
        int curlevel = 0;

        if (guidance_layer_ratio_ < 0.0) {
            
            curlevel = getRandomLevel(mult_);
        } else {
            
            
            
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            const bool belongs_to_guidance_layer =
                distribution(level_generator_) < guidance_layer_ratio_;

            if (belongs_to_guidance_layer) {
                curlevel = guidance_layer_level_ + getRandomLevel(mult_);
            } else {
                do {
                    curlevel = getRandomLevel(mult_);
                } while (curlevel >= guidance_layer_level_);
            }
        }

        if (user_level > 0) {
            curlevel = user_level;
        }

        size_t forced_count = 0;
        auto it = representative_forced_count_.find(attributes);
        if (it != representative_forced_count_.end()) {
            forced_count = it->second;
        }

        if (forced_count < guidance_layer_min_per_attribute_set_ &&
            curlevel < guidance_layer_level_) {
            curlevel = guidance_layer_level_;
        }

        return curlevel;
    }

    size_t representativeVectorDimension() const {
        const size_t dim = getVectorDim();
        if (dim == 0 || data_size_ != dim * sizeof(float)) {
            throw std::runtime_error(
                "mean-based guidance entrypoint selection requires float vectors");
        }
        return dim;
    }

    void initializeRepresentativeMean(
        const AttributeSet& attributes,
        const void *data_point
    ) {
        const size_t dim = representativeVectorDimension();
        const float *new_vector = reinterpret_cast<const float *>(data_point);
        RepresentativeMeanState state;
        state.participant_count = 1;
        state.mean.assign(new_vector, new_vector + dim);
        representative_mean_states_[attributes] = std::move(state);
    }

    
    void updateRepresentativeByOnlineMean(
        tableint internal_id,
        const AttributeSet& attributes,
        const void *data_point
    ) {
        auto state_it = representative_mean_states_.find(attributes);
        auto entry_it = representative_table_.find(attributes);
        if (state_it == representative_mean_states_.end() ||
            entry_it == representative_table_.end()) {
            throw std::runtime_error(
                "missing guidance-layer entrypoint mean state");
        }

        RepresentativeMeanState &state = state_it->second;
        if (state.participant_count == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error(
                "guidance-layer entrypoint mean participant count overflow");
        }

        const size_t dim = representativeVectorDimension();
        if (state.mean.size() != dim) {
            throw std::runtime_error(
                "invalid guidance-layer entrypoint mean dimension");
        }

        const float *new_vector = reinterpret_cast<const float *>(data_point);
        const uint64_t new_count = state.participant_count + 1;
        const double reciprocal_count = 1.0 / static_cast<double>(new_count);
        for (size_t d = 0; d < dim; ++d) {
            state.mean[d] = static_cast<float>(
                static_cast<double>(state.mean[d]) +
                (static_cast<double>(new_vector[d]) -
                 static_cast<double>(state.mean[d])) * reciprocal_count);
        }
        state.participant_count = new_count;

        const dist_t new_node_distance = fstdistfunc_(
            state.mean.data(),
            data_point,
            dist_func_param_);
        const dist_t current_entry_distance = fstdistfunc_(
            state.mean.data(),
            getDataByInternalId(entry_it->second),
            dist_func_param_);
        build_distance_computations_.fetch_add(2, std::memory_order_relaxed);
        guidance_layer_build_distance_computations_.fetch_add(
            2,
            std::memory_order_relaxed);

        if (new_node_distance < current_entry_distance) {
            entry_it->second = internal_id;
        }
    }

    
    void updateRepresentativeBySampledOnlineMean(
        tableint internal_id,
        const AttributeSet& attributes,
        const void *data_point
    ) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        if (distribution(representative_probability_generator_) >=
            representative_sampling_probability_) {
            return;
        }
        updateRepresentativeByOnlineMean(
            internal_id,
            attributes,
            data_point);
    }

    void registerRepresentative(
        tableint internal_id,
        const AttributeSet& attributes,
        int level,
        const void *data_point
    ) {
        if (level < guidance_layer_level_) {
            return;
        }

        std::lock_guard<std::mutex> lock(representative_table_lock_);

        auto ep_it = representative_table_.find(attributes);
        if (ep_it == representative_table_.end()) {
            representative_table_[attributes] = internal_id;
            if (representative_selection_strategy_ != 0) {
                initializeRepresentativeMean(attributes, data_point);
            }
        } else if (representative_selection_strategy_ == 1) {
            updateRepresentativeByOnlineMean(
                internal_id,
                attributes,
                data_point);
        } else if (representative_selection_strategy_ == 2) {
            updateRepresentativeBySampledOnlineMean(
                internal_id,
                attributes,
                data_point);
        }

        size_t &cnt = representative_forced_count_[attributes];
        if (cnt < guidance_layer_min_per_attribute_set_) {
            cnt++;
        }
    }


    
