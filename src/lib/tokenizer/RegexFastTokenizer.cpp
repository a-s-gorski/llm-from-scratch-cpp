#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include "llm_fs/tokenizer/RegexFastTokenizer.h"
#include "trainer/Trainer.h"
#include "encoder/Encoder.h"

#include <iostream>

namespace llm_fs::tokenizer {
    RegexFastTokenizer::RegexFastTokenizer(PatternType pattern_type) : BaseTokenizer() {
        switch(pattern_type) {
            case PatternType::GPT2:
                pattern = boost::regex(pattern_gpt2,
                    boost::regex_constants::icase | boost::regex_constants::optimize);
            break;
            case PatternType::GPT4:
                default:
                    pattern = boost::regex(pattern_gpt4,
                        boost::regex_constants::icase | boost::regex_constants::optimize);
            break;
        }
    }



    void RegexFastTokenizer::train(const std::string text, const unsigned int vocab_size){
        this->vocab_size = vocab_size;
        auto id_list = presplit(text);
        auto trainer = trainer::Trainer();
        auto results = llm_fs::tokenizer::trainer::Trainer::train(id_list, static_cast<int>(vocab_size), static_cast<int>(init_tokens));
        std::tie(this->merges, this->vocab) = calculate_merges_and_vocab(results, static_cast<int>(init_tokens));
        std::cout << "First 20 merges:\n";
        int count = 0;
        for (const auto& merge : this->merges) {
            if (count++ >= 20) break;
            std::cout << "(" << merge.first.first << ", " << merge.first.second << ") -> " << merge.second << "\n";
        }

        // Print first 20 vocabulary entries
        std::cout << "First 20 vocabulary entries:\n";
        count = 0;
        for (const auto& entry : this->vocab) {
            if (count++ >= 20) break;
            std::cout << entry.first << " -> " << entry.second << "\n";
        }

    }

    std::vector<uint32_t>  RegexFastTokenizer::encode(std::string text, const std::optional<std::vector<u_int8_t>> &ids ) {



        return {};
    }

    std::string RegexFastTokenizer::decode(std::vector<uint32_t> tokens) {
        std::vector<std::string> part_bytes;
        for (int idx : tokens) {
            auto vocab_it = vocab.find(idx);
            std::cout << vocab_it->first << vocab_it->second << "\n";
            if (vocab_it != vocab.end()) {
                part_bytes.push_back(vocab_it->second);
            }
        }
        std::string text_bytes;
        for (const auto& bytes : part_bytes) {
            text_bytes += bytes;
        }
        return text_bytes;
    }


    std::vector<uint32_t>  RegexFastTokenizer::encode_efficient(std::string text, const std::optional<std::vector<u_int8_t>> &ids) {
        if (text.empty()) return {};
        std::vector<u_int8_t> ids_processed = {};
        if (ids.has_value()) {
            ids_processed = ids.value();
        }else {
            ids_processed = generateIds(text);
        }
        const std::vector<std::string>& text_chunks = generateTextChunks(text);
        std::vector<int32_t> split_indices = generateSplitIndices(text_chunks, ids_processed.size());

        unsigned long vocab_size = this->merges.size() + this->init_tokens;
        std::vector<int64_t> merges_processed;
        merges_processed.reserve(vocab_size);

        for (const auto&[fst, snd]: this->merges) {
            merges_processed.push_back(fst.first * vocab_size + fst.second);
        }

        auto subencoder = llm_fs::tokenizer::encoder::Encoder();

        std::vector<uint32_t> result = subencoder.tokenize(ids_processed, split_indices, merges_processed, this->init_tokens);


        return result;


    }

    std::vector<int32_t> RegexFastTokenizer::generateSplitIndices(const std::vector<std::string> &text_chunks, int id_processed_size) {
        std::vector<int32_t> split_indices(text_chunks.size() + 1, 0);
        int curr_el = 0;
        for (size_t i = 0; i < text_chunks.size(); ++i) {
            split_indices[i] = curr_el;
            curr_el += text_chunks[i].size();
        }
        split_indices.back() = id_processed_size;
        return  split_indices;
    }



    std::vector<int32_t> RegexFastTokenizer::presplit(const std::string& text) const {
        const std::vector<std::string>& text_chunks = generateTextChunks(text);
        const std::vector<uint8_t>& ids = generateIds(text);

        size_t total_size = 0;
        for (const auto& chunk : text_chunks) {
            total_size += chunk.size() + 1;
        }

        std::vector<int32_t> id_list(total_size, 0);

        size_t position_id_list = 0;
        size_t position_id = 0;

        for (const std::string& chunk : text_chunks) {
            const size_t chunk_length = chunk.size();

            if (position_id + chunk_length > ids.size()) {
                throw std::runtime_error("Mismatch between text chunks and generated IDs");
            }

            std::copy_n(ids.begin() + static_cast<ptrdiff_t>(position_id),
                     static_cast<ptrdiff_t>(chunk_length),
                     id_list.begin() + static_cast<ptrdiff_t>(position_id_list));

            position_id_list += chunk_length + 1;
            position_id += chunk_length;
        }

        return id_list;
    }


    const std::string& RegexFastTokenizer::getPatternGPT2() {
        return pattern_gpt2;
    }

    const std::string& RegexFastTokenizer::getPatternGPT4() {
        return pattern_gpt4;
    }

    std::vector<uint8_t> RegexFastTokenizer::generateIds(const std::string& text){
        return std::vector<uint8_t>{text.begin(), text.end()};
    }

    std::tuple<std::map<std::pair<int, int>, int>, std::map<int, std::string>>
    RegexFastTokenizer::calculate_merges_and_vocab(const std::vector<trainer::Merge>& results, const int init_tokens) {
        std::map<std::pair<int, int>, int> merges;
        std::map<int, std::string> vocab;

        for (const auto& result : results) {
            const std::string token_str(result.token_list.begin(), result.token_list.end());

            if (result.token_id >= init_tokens) {
                merges[{result.first_id, result.second_id}] = result.token_id;
            }
            vocab[result.token_id] = token_str;
        }

        return std::make_tuple(merges, vocab);
    }

    std::vector<std::string> RegexFastTokenizer::generateTextChunks(const std::string& text) const {
        std::vector<std::string> tokens;
        boost::sregex_iterator it(text.begin(), text.end(), pattern);

        for (const boost::sregex_iterator end; it != end; ++it) {
            tokens.push_back(it->str());
        }

        return tokens;
    }

    std::vector<int32_t> RegexFastTokenizer::generateIdsList(const std::string& text){
        return std::vector<int>(text.length(), 0);
    }


}

