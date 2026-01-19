// TCQ4 Channel Calibration Tool
// Computes channel permutations for TCQ4 quantization based on activation statistics.
// Reference: RRS Paper Section 3.2 - "Offline Channel Reordering"
//
// This tool hooks into GGML's computation graph to capture activation statistics
// at each linear layer input, then computes permutations that group outlier channels
// together to improve Runtime Smooth effectiveness.
//
// Usage:
//   tcq4-calibrate -m model.gguf -f calibration.txt -o perms.json
//   llama-quantize --tcq4-perms perms.json model.gguf output.gguf TCQ4_K32

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

static void print_usage(int argc, char ** argv) {
    LOG("\nUsage: %s [options]\n", argv[0]);
    LOG("\n");
    LOG("Compute channel permutations for TCQ4 quantization.\n");
    LOG("\n");
    LOG("Options:\n");
    LOG("  -h, --help              show this help message and exit\n");
    LOG("  -m, --model FILE        model path (required)\n");
    LOG("  -f, --file FILE         calibration text file (required)\n");
    LOG("  -o, --output FILE       output JSON file for permutations (default: perms.json)\n");
    LOG("  -c, --ctx-size N        context size (default: 512)\n");
    LOG("  -b, --batch-size N      batch size (default: 512)\n");
    LOG("  -t, --threads N         number of threads (default: 4)\n");
    LOG("  --chunks N              number of chunks to process (default: all)\n");
    LOG("  --output-freq N         output frequency in chunks (default: 10)\n");
    LOG("  -ngl, --n-gpu-layers N  number of GPU layers (default: 0)\n");
    LOG("  -v, --verbose           verbose output\n");
    LOG("\n");
    LOG("Example:\n");
    LOG("  %s -m model-f16.gguf -f wikitext-2-raw/wiki.train.raw -o perms.json\n", argv[0]);
    LOG("\n");
    (void)argc;
}

// Statistics for a single tensor
struct ChannelStats {
    std::vector<float> channel_max;  // max(|x|) per channel
    int64_t count = 0;               // number of samples
};

// Collector for channel statistics
class ChannelStatsCollector {
public:
    ChannelStatsCollector() = default;

    void set_params(int ctx_size, int output_freq) {
        m_ctx_size = ctx_size;
        m_output_freq = output_freq;
    }

    // Callback for GGML tensor operations
    // Called during forward pass for each MUL_MAT operation
    bool collect_stats(struct ggml_tensor * t, bool ask, void * user_data);

    // Compute permutations from collected statistics
    std::unordered_map<std::string, std::vector<int32_t>> compute_permutations() const;

    // Save permutations to JSON file
    void save_permutations(const std::string & path) const;

    // Get number of tensors tracked
    size_t num_tensors() const { return m_stats.size(); }

    // Get last chunk processed
    int32_t last_chunk() const { return m_last_chunk; }

private:
    std::unordered_map<std::string, ChannelStats> m_stats;
    std::mutex m_mutex;
    int m_ctx_size = 512;
    int m_output_freq = 10;
    int32_t m_last_chunk = 0;
    std::vector<char> m_src1_data;  // buffer for copying GPU data
};

// Filter tensor name to get weight name from src0
static std::string filter_tensor_name(const char * name) {
    if (!name) return "";
    std::string sname(name);
    
    // Remove common suffixes/prefixes
    while (!sname.empty() && (sname.back() == '.' || sname.back() == '_')) {
        sname.pop_back();
    }
    
    return sname;
}

bool ChannelStatsCollector::collect_stats(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;

    // We only care about MUL_MAT operations (linear layers)
    if (t->op != GGML_OP_MUL_MAT && t->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    const struct ggml_tensor * src0 = t->src[0];  // weights
    const struct ggml_tensor * src1 = t->src[1];  // activations
    
    std::string wname = filter_tensor_name(src0->name);

    // When ask is true, scheduler wants to know if we're interested
    if (ask) {
        if (t->op == GGML_OP_MUL_MAT_ID) return true;
        if (t->op != GGML_OP_MUL_MAT) return false;
        // Only collect for block tensors and optionally output
        if (src1->ne[1] < 16 || src1->type != GGML_TYPE_F32) return false;
        if (wname.substr(0, 4) != "blk." && wname != "output.weight") return false;
        return true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    // Copy data from GPU if needed
    const bool is_host = ggml_backend_buffer_is_host(src1->buffer);
    if (!is_host) {
        const size_t src1_nbytes = ggml_nbytes(src1);
        m_src1_data.resize(src1_nbytes);
        ggml_backend_tensor_get(src1, m_src1_data.data(), 0, src1_nbytes);
    }

    const char * data = is_host ? (const char *)src1->data : m_src1_data.data();
    GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));

    // Handle MUL_MAT_ID (MoE)
    if (t->op == GGML_OP_MUL_MAT_ID) {
        // For MoE, we still track but use simplified handling
        // The experts share input dimension, so we can aggregate
        auto & e = m_stats[wname];
        if (e.channel_max.empty()) {
            e.channel_max.resize(src1->ne[0], 0.0f);
        }
        
        for (int64_t row = 0; row < ggml_nrows(src1); ++row) {
            const float * x = (const float *)(data + row * src1->nb[1]);
            for (int64_t j = 0; j < src1->ne[0]; ++j) {
                float abs_val = std::fabs(x[j]);
                e.channel_max[j] = std::max(e.channel_max[j], abs_val);
            }
            e.count++;
        }
    } else {
        // Regular MUL_MAT
        auto & e = m_stats[wname];
        const int64_t n_channels = src1->ne[0];

        if (e.channel_max.empty()) {
            e.channel_max.resize(n_channels, 0.0f);
        } else if ((int64_t)e.channel_max.size() != n_channels) {
            LOG_ERR("%s: inconsistent size for %s (%zu vs %lld)\n", 
                    __func__, wname.c_str(), e.channel_max.size(), (long long)n_channels);
            return true;
        }

        // Iterate over all rows and update channel max
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                for (int64_t row = 0; row < src1->ne[1]; ++row) {
                    const float * x = (const float *)(data + row * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3]);
                    for (int64_t j = 0; j < n_channels; ++j) {
                        float abs_val = std::fabs(x[j]);
                        e.channel_max[j] = std::max(e.channel_max[j], abs_val);
                    }
                    e.count++;
                }
            }
        }

        // Progress tracking
        const int32_t n_chunk = e.count / m_ctx_size;
        if (n_chunk > m_last_chunk) {
            m_last_chunk = n_chunk;
            if (m_last_chunk % m_output_freq == 0) {
                LOG("  [chunk %d] %zu tensors tracked\n", m_last_chunk, m_stats.size());
            }
        }
    }

    return true;
}

std::unordered_map<std::string, std::vector<int32_t>> ChannelStatsCollector::compute_permutations() const {
    std::unordered_map<std::string, std::vector<int32_t>> perms;

    for (const auto & kv : m_stats) {
        const std::string & name = kv.first;
        const ChannelStats & stats = kv.second;
        const int64_t K = (int64_t)stats.channel_max.size();

        if (K == 0) continue;

        // Create index array
        std::vector<int32_t> perm(K);
        for (int32_t i = 0; i < K; ++i) {
            perm[i] = i;
        }

        // Sort by channel_max descending (outliers first)
        std::sort(perm.begin(), perm.end(), [&stats](int32_t a, int32_t b) {
            return stats.channel_max[a] > stats.channel_max[b];
        });

        perms[name] = std::move(perm);
    }

    return perms;
}

void ChannelStatsCollector::save_permutations(const std::string & path) const {
    auto perms = compute_permutations();

    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERR("Failed to open output file: %s\n", path.c_str());
        return;
    }

    out << "{\n";
    out << "  \"version\": 1,\n";
    out << "  \"description\": \"TCQ4 channel permutations from calibration\",\n";
    out << "  \"reorder_enabled\": true,\n";
    out << "  \"permutations\": {\n";

    size_t idx = 0;
    for (const auto & kv : perms) {
        const std::string & name = kv.first;
        const std::vector<int32_t> & perm = kv.second;

        out << "    \"" << name << "\": [";
        for (size_t i = 0; i < perm.size(); ++i) {
            if (i > 0) out << ", ";
            out << perm[i];
        }
        out << "]";
        if (++idx < perms.size()) out << ",";
        out << "\n";
    }

    out << "  }\n";
    out << "}\n";

    out.close();
    LOG("Saved %zu permutations to %s\n", perms.size(), path.c_str());
}

// Global collector instance (needed for callback)
static ChannelStatsCollector g_collector;

// Callback wrapper for GGML
static bool tcq4_collect_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    return g_collector.collect_stats(t, ask, user_data);
}

// Process calibration data through the model
static bool compute_calibration(
    const std::string & model_path,
    const std::string & calib_path,
    const std::string & output_path,
    int ctx_size,
    int batch_size,
    int n_threads,
    int n_gpu_layers,
    int max_chunks,
    int output_freq,
    bool verbose
) {
    // Load calibration text
    std::ifstream calib_file(calib_path);
    if (!calib_file.is_open()) {
        LOG_ERR("Failed to open calibration file: %s\n", calib_path.c_str());
        return false;
    }

    std::string calib_text((std::istreambuf_iterator<char>(calib_file)),
                            std::istreambuf_iterator<char>());
    calib_file.close();

    if (calib_text.empty()) {
        LOG_ERR("Calibration file is empty: %s\n", calib_path.c_str());
        return false;
    }

    LOG("Loaded calibration text: %zu bytes\n", calib_text.size());

    // Initialize llama backend
    llama_backend_init();

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        LOG_ERR("Failed to load model: %s\n", model_path.c_str());
        llama_backend_free();
        return false;
    }

    LOG("Model loaded: %s\n", model_path.c_str());

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = ctx_size;
    ctx_params.n_batch = batch_size;
    ctx_params.n_ubatch = batch_size;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.cb_eval = tcq4_collect_callback;
    ctx_params.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        llama_model_free(model);
        llama_backend_free();
        return false;
    }

    // Set up collector
    g_collector.set_params(ctx_size, output_freq);

    // Tokenize calibration text
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(calib_text.size() + 1);
    int n_tokens = llama_tokenize(vocab, calib_text.c_str(), calib_text.size(), 
                                   tokens.data(), tokens.size(), true, false);
    if (n_tokens < 0) {
        LOG_ERR("Failed to tokenize calibration text\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return false;
    }
    tokens.resize(n_tokens);

    LOG("Tokenized: %d tokens\n", n_tokens);

    // Process tokens in chunks
    const int n_chunk = (n_tokens + ctx_size - 1) / ctx_size;
    const int chunks_to_process = (max_chunks > 0) ? std::min(max_chunks, n_chunk) : n_chunk;

    LOG("Processing %d chunks (of %d total)...\n", chunks_to_process, n_chunk);

    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    for (int i = 0; i < chunks_to_process; ++i) {
        const int start = i * ctx_size;
        const int end = std::min(start + ctx_size, n_tokens);
        const int chunk_tokens = end - start;

        if (chunk_tokens <= 0) break;

        // Clear batch
        batch.n_tokens = 0;

        // Add tokens to batch
        for (int j = 0; j < chunk_tokens; ++j) {
            batch.token[batch.n_tokens] = tokens[start + j];
            batch.pos[batch.n_tokens] = j;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = (j == chunk_tokens - 1);
            batch.n_tokens++;
        }

        // Decode (this triggers the callbacks)
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode chunk %d\n", i);
            break;
        }

        // Clear KV cache for next chunk
        llama_memory_clear(llama_get_memory(ctx), true);

        if (verbose && (i + 1) % output_freq == 0) {
            LOG("  Processed chunk %d/%d\n", i + 1, chunks_to_process);
        }
    }

    llama_batch_free(batch);

    LOG("\nCalibration complete. Collected stats for %zu tensors\n", g_collector.num_tensors());

    // Save permutations
    g_collector.save_permutations(output_path);

    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return true;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string calib_path;
    std::string output_path = "perms.json";
    int ctx_size = 512;
    int batch_size = 512;
    int n_threads = 4;
    int n_gpu_layers = 0;
    int max_chunks = 0;
    int output_freq = 10;
    bool verbose = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (++i < argc) model_path = argv[i];
        } else if (arg == "-f" || arg == "--file") {
            if (++i < argc) calib_path = argv[i];
        } else if (arg == "-o" || arg == "--output") {
            if (++i < argc) output_path = argv[i];
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (++i < argc) ctx_size = std::stoi(argv[i]);
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i < argc) batch_size = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i < argc) n_threads = std::stoi(argv[i]);
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i < argc) n_gpu_layers = std::stoi(argv[i]);
        } else if (arg == "--chunks") {
            if (++i < argc) max_chunks = std::stoi(argv[i]);
        } else if (arg == "--output-freq") {
            if (++i < argc) output_freq = std::stoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else {
            LOG_ERR("Unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            return 1;
        }
    }

    // Validate required arguments
    if (model_path.empty()) {
        LOG_ERR("Model path is required (-m)\n");
        print_usage(argc, argv);
        return 1;
    }
    if (calib_path.empty()) {
        LOG_ERR("Calibration file is required (-f)\n");
        print_usage(argc, argv);
        return 1;
    }

    LOG("\n");
    LOG("TCQ4 Channel Calibration\n");
    LOG("========================\n");
    LOG("Model:       %s\n", model_path.c_str());
    LOG("Calibration: %s\n", calib_path.c_str());
    LOG("Output:      %s\n", output_path.c_str());
    LOG("Context:     %d\n", ctx_size);
    LOG("Batch:       %d\n", batch_size);
    LOG("Threads:     %d\n", n_threads);
    LOG("GPU layers:  %d\n", n_gpu_layers);
    LOG("\n");

    bool success = compute_calibration(
        model_path, calib_path, output_path,
        ctx_size, batch_size, n_threads, n_gpu_layers,
        max_chunks, output_freq, verbose
    );

    if (success) {
        LOG("\nTo quantize with channel reordering:\n");
        LOG("  llama-quantize --tcq4-perms %s %s output.gguf TCQ4_K32\n", 
            output_path.c_str(), model_path.c_str());
    }

    return success ? 0 : 1;
}