#include "model_detection.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "ggml/include/gguf.h"
#include "crow.h"
#include "logging.h"

// Helper function to inspect tensor keys from safetensors file
static std::vector<std::string> getSafetensorsTensorKeys(const std::string& model_path) {
    std::vector<std::string> tensor_keys;

    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        return tensor_keys;
    }

    // Read the header size (first 8 bytes, little-endian)
    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    if (header_size == 0 || header_size > 100000000) {  // Sanity check (100MB max header)
        return tensor_keys;
    }

    // Read the JSON header
    std::string header_json(header_size, '\0');
    file.read(&header_json[0], header_size);
    file.close();

    // Parse JSON to get tensor keys
    auto header = crow::json::load(header_json);
    if (header) {
        for (const auto& key : header.keys()) {
            std::string key_str(key);
            // Skip metadata key
            if (key_str != "__metadata__") {
                tensor_keys.push_back(key_str);
            }
        }
    }

    return tensor_keys;
}

// Helper function to inspect tensor keys from GGUF file
static std::vector<std::string> getGGUFTensorKeys(const std::string& model_path) {
    std::vector<std::string> tensor_keys;

    gguf_init_params params;
    params.no_alloc = true;  // We don't need to allocate tensor data
    params.ctx = nullptr;

    gguf_context* ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        LOG_ERROR("Failed to open GGUF file for inspection: %s", model_path.c_str());
        return tensor_keys;
    }

    // Extract tensor keys from GGUF
    int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; i++) {
        const char* tensor_name = gguf_get_tensor_name(ctx, i);
        if (tensor_name) {
            tensor_keys.push_back(std::string(tensor_name));
        }
    }

    gguf_free(ctx);
    return tensor_keys;
}

// Helper function to determine model type from tensor keys
// Returns: "clip_l", "clip_g", "t5xxl", "llm", or "vae" (default)
std::string inferModelTypeFromTensorKeys(const std::vector<std::string>& tensor_keys) {
    if (tensor_keys.empty()) {
        return "vae";  // Default to VAE if we can't determine
    }

    bool has_text_model = false;
    bool has_text_projection = false;
    bool has_position_ids = false;
    bool has_self_attention = false;
    bool has_dense_relu_dense = false;
    bool has_llm_token_embedding = false;
    bool has_llm_attention = false;
    bool has_llm_mlp = false;
    bool has_llm_output_norm = false;
    bool has_llm_qk_norm = false;
    bool has_llm_moe_mlp = false;
    bool has_llm_moe_router = false;

    // Count transformer layers to distinguish CLIP-L (12 layers) from CLIP-G (32 layers)
    int max_layer_number = -1;

    // Check tensor keys for patterns
    for (const std::string& name : tensor_keys) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);

        // CLIP text model indicators
        if (name_lower.find("text_model") != std::string::npos ||
            name_lower.find("transformer.") != std::string::npos) {
            has_text_model = true;
        }
        if (name_lower.find("text_projection") != std::string::npos) {
            has_text_projection = true;
        }
        if (name_lower.find("position_ids") != std::string::npos) {
            has_position_ids = true;
        }

        // T5 model indicators
        if (name_lower.find("selfattention") != std::string::npos) {
            has_self_attention = true;
        }
        if (name_lower.find("denserelu") != std::string::npos) {
            has_dense_relu_dense = true;
        }

        // LLM model indicators. Support both raw GGUF naming and converted safetensors naming.
        if (name_lower.find("token_embd.weight") != std::string::npos ||
            name_lower.find("embed_tokens.weight") != std::string::npos) {
            has_llm_token_embedding = true;
        }
        if ((name_lower.find("blk.") != std::string::npos &&
             (name_lower.find("attn_q.weight") != std::string::npos ||
              name_lower.find("attn_k.weight") != std::string::npos ||
              name_lower.find("attn_v.weight") != std::string::npos ||
              name_lower.find("attn_output.weight") != std::string::npos)) ||
            (name_lower.find("model.layers.") != std::string::npos &&
             (name_lower.find("self_attn.q_proj.weight") != std::string::npos ||
              name_lower.find("self_attn.k_proj.weight") != std::string::npos ||
              name_lower.find("self_attn.v_proj.weight") != std::string::npos ||
              name_lower.find("self_attn.o_proj.weight") != std::string::npos))) {
            has_llm_attention = true;
        }
        if ((name_lower.find("blk.") != std::string::npos &&
             (name_lower.find("ffn_gate.weight") != std::string::npos ||
              name_lower.find("ffn_up.weight") != std::string::npos ||
              name_lower.find("ffn_down.weight") != std::string::npos)) ||
            (name_lower.find("model.layers.") != std::string::npos &&
             (name_lower.find("mlp.gate_proj.weight") != std::string::npos ||
              name_lower.find("mlp.up_proj.weight") != std::string::npos ||
              name_lower.find("mlp.down_proj.weight") != std::string::npos))) {
            has_llm_mlp = true;
        }
        if (name_lower.find("output_norm.weight") != std::string::npos ||
            name_lower.find("model.norm.weight") != std::string::npos) {
            has_llm_output_norm = true;
        }
        if (name_lower.find("attn_q_norm.weight") != std::string::npos ||
            name_lower.find("attn_k_norm.weight") != std::string::npos ||
            name_lower.find("self_attn.q_norm.weight") != std::string::npos ||
            name_lower.find("self_attn.k_norm.weight") != std::string::npos) {
            has_llm_qk_norm = true;
        }
        if ((name_lower.find("blk.") != std::string::npos &&
             (name_lower.find("ffn_gate_exps.weight") != std::string::npos ||
              name_lower.find("ffn_up_exps.weight") != std::string::npos ||
              name_lower.find("ffn_down_exps.weight") != std::string::npos)) ||
            (name_lower.find("model.layers.") != std::string::npos &&
             (name_lower.find("mlp.experts.gate_proj") != std::string::npos ||
              name_lower.find("mlp.experts.up_proj") != std::string::npos ||
              name_lower.find("mlp.experts.down_proj") != std::string::npos))) {
            has_llm_moe_mlp = true;
        }
        if (name_lower.find("ffn_gate_inp.weight") != std::string::npos ||
            name_lower.find("mlp.router") != std::string::npos ||
            name_lower.find("mlp.gate.weight") != std::string::npos) {
            has_llm_moe_router = true;
        }

        // Extract layer numbers from tensor names
        // Look for patterns like "layers.11", "layer.31", "blocks.5", etc.
        if (name_lower.find("layer") != std::string::npos || name_lower.find("block") != std::string::npos) {
            // Split by dots and look for numeric parts
            size_t pos = 0;
            while (pos < name_lower.length()) {
                if (std::isdigit(name_lower[pos])) {
                    int layer_num = 0;
                    while (pos < name_lower.length() && std::isdigit(name_lower[pos])) {
                        layer_num = layer_num * 10 + (name_lower[pos] - '0');
                        pos++;
                    }
                    if (layer_num > max_layer_number) {
                        max_layer_number = layer_num;
                    }
                } else {
                    pos++;
                }
            }
        }
    }

    // Check for T5 model (has SelfAttention and DenseReluDense patterns)
    if (has_self_attention && has_dense_relu_dense) {
        LOG_DEBUG("Detected T5 model");
        return "t5xxl";
    }

    // Qwen3 and similar LLMs expose a transformer block structure with token embeddings,
    // attention projections, MLP projections, and a final output norm.
    bool mlp_present = has_llm_mlp || has_llm_moe_mlp;
    if ((has_llm_token_embedding && has_llm_attention && mlp_present) ||
        (has_llm_attention && mlp_present && has_llm_output_norm) ||
        (has_llm_attention && mlp_present && has_llm_qk_norm) ||
        (has_llm_attention && has_llm_moe_mlp && has_llm_moe_router)) {
        LOG_DEBUG("Detected LLM model");
        return "llm";
    }

    // If it's a CLIP model (has text model indicators)
    if (has_text_model || has_text_projection || has_position_ids) {
        // Distinguish between CLIP-L and CLIP-G based on layer count
        // CLIP-L: 12 transformer layers (0-11, max = 11)
        // CLIP-G: 32 transformer layers (0-31, max = 31)

        LOG_DEBUG("Detected CLIP model with max layer number: %d", max_layer_number);

        if (max_layer_number >= 20) {
            // CLIP-G has 32 layers (0-31)
            return "clip_g";
        } else {
            // CLIP-L has 12 layers (0-11), or couldn't detect layer count
            return "clip_l";
        }
    }

    // Not a CLIP or T5 model, assume it's a VAE
    return "vae";
}

// Helper function to detect file format by reading magic header
// Returns: "gguf", "safetensors", or "unknown"
std::string detectModelFormat(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        return "unknown";
    }

    // Read first 8 bytes to check magic headers
    char header[8] = {0};
    file.read(header, 8);
    file.close();

    // Check for GGUF magic (first 4 bytes: "GGUF")
    if (header[0] == 'G' && header[1] == 'G' && header[2] == 'U' && header[3] == 'F') {
        return "gguf";
    }

    // Check for safetensors format (first 8 bytes should be a valid header size)
    // The header size is stored as little-endian uint64
    uint64_t header_size = 0;
    memcpy(&header_size, header, sizeof(uint64_t));

    // Sanity check: header size should be reasonable (not too small, not too large)
    if (header_size > 8 && header_size < 100000000) {  // Between 8 bytes and 100MB
        return "safetensors";
    }

    return "unknown";
}

// Extract tensor keys from a model file based on its format
// Supports both GGUF and safetensors formats
std::vector<std::string> extractTensorKeys(const std::string& model_path, const std::string& format) {
    if (format == "safetensors") {
        return getSafetensorsTensorKeys(model_path);
    } else if (format == "gguf") {
        return getGGUFTensorKeys(model_path);
    } else {
        return std::vector<std::string>();
    }
}

// Helper function to inspect a model file and determine its type
// Supports both GGUF and safetensors formats
// Returns: "vae", "clip_l", "clip_g", "t5xxl", "llm", or "unknown"
std::string inspectModelType(const std::string& model_path) {
    // First, detect the file type
    std::string format = detectModelFormat(model_path);
    if (format == "unknown") {
        LOG_ERROR("Unknown or unsupported file format for: %s", model_path.c_str());
        return "unknown";
    }

    LOG_DEBUG("Inspecting %s file: %s", format.c_str(), model_path.c_str());

    // Extract tensor keys from the respective function
    std::vector<std::string> tensor_keys = extractTensorKeys(model_path, format);
    if (tensor_keys.empty()) {
        LOG_ERROR("Failed to read %s file: %s", format.c_str(), model_path.c_str());
        return "unknown";
    }

    LOG_DEBUG("Found %zu tensors in %s file", tensor_keys.size(), format.c_str());

    // Run inferModelType on the keys
    std::string model_type = inferModelTypeFromTensorKeys(tensor_keys);
    LOG_INFO("Inspected %s model %s: detected type = %s", format.c_str(), model_path.c_str(), model_type.c_str());
    return model_type;
}
