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

    static void writeAttributeSet(
        std::ostream& output,
        const AttributeSet& attributes) {
        if (attributes.size() > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("attribute set is too large to serialize");
        }
        const uint32_t count = static_cast<uint32_t>(attributes.size());
        writeBinaryPOD(output, count);
        for (std::int32_t attribute : attributes.values()) {
            writeBinaryPOD(output, attribute);
        }
    }

    static AttributeSet readAttributeSet(std::istream& input) {
        uint32_t count = 0;
        readBinaryPOD(input, count);
        if (!input || count > 1048576U) {
            throw std::runtime_error("invalid serialized attribute count");
        }
        std::vector<std::int32_t> values(count);
        for (uint32_t index = 0; index < count; ++index) {
            readBinaryPOD(input, values[index]);
        }
        if (!input) {
            throw std::runtime_error("truncated serialized attribute set");
        }
        return AttributeSet(std::move(values));
    }

    static size_t serializedAttributeSetSize(const AttributeSet& attributes) {
        return sizeof(uint32_t) + attributes.size() * sizeof(std::int32_t);
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(uint64_t);
        size += sizeof(uint32_t);
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
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

        size += sizeof(uint32_t);
        size += sizeof(size_t);
        for (size_t i = 0; i < cur_element_count; ++i) {
            size += serializedAttributeSetSize(element_attributes_[i]);
        }

        size += sizeof(guidance_layer_level_);
        size += sizeof(guidance_layer_min_per_attribute_set_);
        size += sizeof(guidance_rewrite_topk_);

        size_t entrypoint_count = representative_table_.size();
        size += sizeof(entrypoint_count);

        for (const auto& kv : representative_table_) {
            size += serializedAttributeSetSize(kv.first);
            size += sizeof(tableint);
        }

        size += sizeof(uint32_t);  
        size += sizeof(acorn_build_mode_);
        size += sizeof(acorn_gamma_);
        size += sizeof(acorn_m_beta_);
        size += sizeof(uint32_t);  
        size += sizeof(guidance_layer_ratio_);
        size += sizeof(uint32_t);  
        size += sizeof(guidance_layer_assignment_version_);

        size += sizeof(uint32_t);  
        size += sizeof(representative_selection_strategy_);
        size_t mean_state_count = representative_mean_states_.size();
        size += sizeof(mean_state_count);
        for (const auto& kv : representative_mean_states_) {
            size += serializedAttributeSetSize(kv.first);
            size += sizeof(uint64_t);
            size += sizeof(size_t);
            size += kv.second.mean.size() * sizeof(float);
        }

        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("Cannot open index file for writing: " + location);
        }
        std::streampos position;

        const uint64_t index_magic = 0x52474E494E445831ULL;
        const uint32_t index_version = 1U;
        writeBinaryPOD(output, index_magic);
        writeBinaryPOD(output, index_version);

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
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

        const uint32_t attribute_block_magic = 0x41545452U;
        writeBinaryPOD(output, attribute_block_magic);
        const size_t attribute_record_count = cur_element_count.load();
        writeBinaryPOD(output, attribute_record_count);
        for (size_t i = 0; i < attribute_record_count; ++i) {
            writeAttributeSet(output, element_attributes_[i]);
        }

        writeBinaryPOD(output, guidance_layer_level_);
        writeBinaryPOD(output, guidance_layer_min_per_attribute_set_);
        writeBinaryPOD(output, guidance_rewrite_topk_);

        size_t entrypoint_count = representative_table_.size();
        writeBinaryPOD(output, entrypoint_count);

        for (const auto& kv : representative_table_) {
            writeAttributeSet(output, kv.first);
            writeBinaryPOD(output, kv.second);
        }

        const uint32_t acorn_config_magic = 0x41434F52U;  
        writeBinaryPOD(output, acorn_config_magic);
        writeBinaryPOD(output, acorn_build_mode_);
        writeBinaryPOD(output, acorn_gamma_);
        writeBinaryPOD(output, acorn_m_beta_);

        const uint32_t guidance_ratio_magic = 0x43505231U;  
        writeBinaryPOD(output, guidance_ratio_magic);
        writeBinaryPOD(output, guidance_layer_ratio_);

        const uint32_t guidance_assignment_magic = 0x43505632U;  
        writeBinaryPOD(output, guidance_assignment_magic);
        writeBinaryPOD(output, guidance_layer_assignment_version_);

        const uint32_t guidance_entrypoint_strategy_magic = 0x43455331U;  
        writeBinaryPOD(output, guidance_entrypoint_strategy_magic);
        writeBinaryPOD(output, representative_selection_strategy_);
        size_t mean_state_count = representative_mean_states_.size();
        writeBinaryPOD(output, mean_state_count);
        for (const auto& kv : representative_mean_states_) {
            writeAttributeSet(output, kv.first);
            writeBinaryPOD(output, kv.second.participant_count);
            const size_t mean_dimension = kv.second.mean.size();
            writeBinaryPOD(output, mean_dimension);
            for (float value : kv.second.mean) {
                writeBinaryPOD(output, value);
            }
        }

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
        guidance_layer_ratio_ = -1.0;
        guidance_layer_assignment_version_ = 2;
        representative_selection_strategy_ = 0;
        representative_mean_states_.clear();
        
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        uint64_t index_magic = 0;
        uint32_t index_version = 0;
        readBinaryPOD(input, index_magic);
        readBinaryPOD(input, index_version);
        if (index_magic != 0x52474E494E445831ULL || index_version != 1U) {
            throw std::runtime_error(
                "Unsupported index format; rebuild the index with this RGN version");
        }

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

        const size_t expected_level0_links =
            maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        if (offsetLevel0_ != 0 ||
            offsetData_ != expected_level0_links ||
            label_offset_ != expected_level0_links + data_size_ ||
            size_data_per_element_ != label_offset_ + sizeof(labeltype)) {
            throw std::runtime_error(
                "Index vector layout does not match the current vector space");
        }

        auto pos = input.tellg();

        
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

        
        
        if (input.tellg() > total_filesize || input.tellg() < 0)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        

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

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        element_attributes_ = std::vector<AttributeSet>(max_elements);
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

        uint32_t attribute_block_magic = 0;
        size_t attribute_record_count = 0;
        readBinaryPOD(input, attribute_block_magic);
        readBinaryPOD(input, attribute_record_count);
        if (attribute_block_magic != 0x41545452U ||
            attribute_record_count != cur_element_count.load()) {
            throw std::runtime_error("Invalid structured-attribute block");
        }
        for (size_t i = 0; i < attribute_record_count; ++i) {
            element_attributes_[i] = readAttributeSet(input);
        }

        if (!input) {
            throw std::runtime_error("Index seems to be corrupted or unsupported");
        }
        if (input.peek() != EOF) {
            readBinaryPOD(input, guidance_layer_level_);
            readBinaryPOD(input, guidance_layer_min_per_attribute_set_);
            readBinaryPOD(input, guidance_rewrite_topk_);

            size_t entrypoint_count = 0;
            readBinaryPOD(input, entrypoint_count);

            representative_table_.clear();
            for (size_t i = 0; i < entrypoint_count; ++i) {
                tableint entry_id;
                AttributeSet attributes = readAttributeSet(input);
                readBinaryPOD(input, entry_id);
                if (entry_id >= cur_element_count.load() ||
                    element_attributes_[entry_id] != attributes) {
                    throw std::runtime_error("Invalid guidance entrypoint record");
                }
                representative_table_[std::move(attributes)] = entry_id;
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
            uint32_t guidance_ratio_magic = 0;
            readBinaryPOD(input, guidance_ratio_magic);
            if (guidance_ratio_magic != 0x43505231U) {
                throw std::runtime_error("Invalid guidance-layer ratio block");
            }
            readBinaryPOD(input, guidance_layer_ratio_);
            if (!std::isfinite(guidance_layer_ratio_) ||
                (guidance_layer_ratio_ != -1.0 &&
                    (guidance_layer_ratio_ < 0.0 || guidance_layer_ratio_ > 1.0))) {
                throw std::runtime_error("Invalid saved guidance-layer ratio");
            }

            
            
            guidance_layer_assignment_version_ = 1;
        }
        if (input.peek() != EOF) {
            uint32_t guidance_assignment_magic = 0;
            readBinaryPOD(input, guidance_assignment_magic);
            if (guidance_assignment_magic != 0x43505632U) {
                throw std::runtime_error(
                    "Invalid guidance-layer assignment version block");
            }
            readBinaryPOD(input, guidance_layer_assignment_version_);
            if (guidance_layer_assignment_version_ != 2) {
                throw std::runtime_error(
                    "Unsupported guidance-layer assignment version");
            }
        }
        if (input.peek() != EOF) {
            uint32_t guidance_entrypoint_strategy_magic = 0;
            readBinaryPOD(input, guidance_entrypoint_strategy_magic);
            if (guidance_entrypoint_strategy_magic != 0x43455331U) {
                throw std::runtime_error(
                    "Invalid guidance-layer entrypoint strategy block");
            }
            readBinaryPOD(
                input,
                representative_selection_strategy_);
            if (representative_selection_strategy_ > 2) {
                throw std::runtime_error(
                    "Unsupported guidance-layer entrypoint selection strategy");
            }

            size_t mean_state_count = 0;
            readBinaryPOD(input, mean_state_count);
            if (mean_state_count > representative_table_.size()) {
                throw std::runtime_error(
                    "Invalid guidance-layer entrypoint mean-state count");
            }

            const size_t expected_dimension = getVectorDim();
            representative_mean_states_.clear();
            for (size_t i = 0; i < mean_state_count; ++i) {
                RepresentativeMeanState state;
                size_t mean_dimension = 0;
                AttributeSet attributes = readAttributeSet(input);
                readBinaryPOD(input, state.participant_count);
                readBinaryPOD(input, mean_dimension);
                if (state.participant_count == 0 ||
                    mean_dimension != expected_dimension ||
                    representative_table_.find(attributes) ==
                        representative_table_.end()) {
                    throw std::runtime_error(
                        "Invalid guidance-layer entrypoint mean state");
                }
                state.mean.resize(mean_dimension);
                for (size_t d = 0; d < mean_dimension; ++d) {
                    readBinaryPOD(input, state.mean[d]);
                }
                representative_mean_states_.emplace(
                    std::move(attributes),
                    std::move(state));
            }

            if (representative_selection_strategy_ == 0 &&
                !representative_mean_states_.empty()) {
                throw std::runtime_error(
                    "First-node entrypoint strategy must not contain mean states");
            }
            if (representative_selection_strategy_ != 0 &&
                representative_mean_states_.size() !=
                    representative_table_.size()) {
                throw std::runtime_error(
                    "Mean-based entrypoint strategy is missing attribute-set state");
            }
        }
        if (maxM_ != M_ * acorn_gamma_ || maxM0_ != 2 * maxM_ ||
            (acorn_build_mode_ == 1 && acorn_m_beta_ != maxM_)) {
            throw std::runtime_error(
                "This build only supports ACORN-1 or uncompressed ACORN-gamma indexes");
        }
        
        if (input.peek() != EOF) {
            throw std::runtime_error("Index seems to be corrupted or unsupported");
        }
        input.close();

        return;
    }


