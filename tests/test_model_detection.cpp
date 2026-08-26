#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "../include/model_detection.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSER_EQ(actual, expected, test_name)                            \
    do {                                                                 \
        if ((actual) != (expected)) {                                    \
            std::cout << "FAILED: " << (test_name) << "\n";              \
            std::cout << "  expected: " << (expected) << "\n";           \
            std::cout << "  actual:   " << (actual) << "\n";             \
            tests_failed++;                                              \
        } else {                                                         \
            std::cout << "PASSED\n";                                     \
            tests_passed++;                                              \
        }                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Helper: build GPT-OSS 20B GGUF tensor keys (llama.cpp naming)
// ---------------------------------------------------------------------------
static std::vector<std::string> make_gpt_oss_gguf_keys() {
    std::vector<std::string> keys;
    keys.push_back("token_embd.weight");
    for (int layer = 0; layer < 24; layer++) {
        keys.push_back("blk." + std::to_string(layer) + ".attn_norm.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_post_norm.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_q.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_k.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_v.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_output.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_sinks.weight");
        keys.push_back("blk." + std::to_string(layer) + ".ffn_gate_inp.weight");
        for (int expert = 0; expert < 32; expert++) {
            keys.push_back("blk." + std::to_string(layer) + ".ffn_gate_exps." + std::to_string(expert) + ".weight");
            keys.push_back("blk." + std::to_string(layer) + ".ffn_up_exps." + std::to_string(expert) + ".weight");
            keys.push_back("blk." + std::to_string(layer) + ".ffn_down_exps." + std::to_string(expert) + ".weight");
        }
    }
    keys.push_back("output_norm.weight");
    keys.push_back("output.weight");
    return keys;
}

// ---------------------------------------------------------------------------
// Helper: build GPT-OSS 20B safetensors keys (converted naming)
// ---------------------------------------------------------------------------
static std::vector<std::string> make_gpt_oss_safetensors_keys() {
    std::vector<std::string> keys;
    keys.push_back("model.embed_tokens.weight");
    for (int layer = 0; layer < 24; layer++) {
        keys.push_back("model.layers." + std::to_string(layer) + ".input_layernorm.weight");
        keys.push_back("model.layers." + std::to_string(layer) + ".post_attention_norm.weight");
        keys.push_back("model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight");
        keys.push_back("model.layers." + std::to_string(layer) + ".self_attn.k_proj.weight");
        keys.push_back("model.layers." + std::to_string(layer) + ".self_attn.v_proj.weight");
        keys.push_back("model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight");
        keys.push_back("model.layers." + std::to_string(layer) + ".self_attn.sinks");
        keys.push_back("model.layers." + std::to_string(layer) + ".mlp.router.weight");
        for (int expert = 0; expert < 32; expert++) {
            keys.push_back("model.layers." + std::to_string(layer) + ".mlp.experts.gate_proj.weight");
            keys.push_back("model.layers." + std::to_string(layer) + ".mlp.experts.up_proj.weight");
            keys.push_back("model.layers." + std::to_string(layer) + ".mlp.experts.down_proj.weight");
        }
    }
    keys.push_back("model.norm.weight");
    keys.push_back("lm_head.weight");
    return keys;
}

// ---------------------------------------------------------------------------
// Helper: standard dense LLM (Qwen-style) GGUF keys
// ---------------------------------------------------------------------------
static std::vector<std::string> make_standard_llm_gguf_keys() {
    std::vector<std::string> keys;
    keys.push_back("token_embd.weight");
    for (int layer = 0; layer < 28; layer++) {
        keys.push_back("blk." + std::to_string(layer) + ".attn_norm.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_q.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_k.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_v.weight");
        keys.push_back("blk." + std::to_string(layer) + ".attn_output.weight");
        keys.push_back("blk." + std::to_string(layer) + ".ffn_gate.weight");
        keys.push_back("blk." + std::to_string(layer) + ".ffn_up.weight");
        keys.push_back("blk." + std::to_string(layer) + ".ffn_down.weight");
    }
    keys.push_back("output_norm.weight");
    keys.push_back("output.weight");
    return keys;
}

// ---------------------------------------------------------------------------
// Helper: CLIP-G safetensors keys
// ---------------------------------------------------------------------------
static std::vector<std::string> make_clip_g_keys() {
    std::vector<std::string> keys;
    keys.push_back("text_model.embeddings.position_ids");
    for (int layer = 0; layer < 32; layer++) {
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".self_attn.q_proj.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".self_attn.k_proj.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".self_attn.v_proj.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".self_attn.out_proj.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".mlp.fc1.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".mlp.fc2.weight");
    }
    keys.push_back("text_model.text_projection.weight");
    return keys;
}

// ---------------------------------------------------------------------------
// Helper: CLIP-L safetensors keys
// ---------------------------------------------------------------------------
static std::vector<std::string> make_clip_l_keys() {
    std::vector<std::string> keys;
    keys.push_back("text_model.embeddings.position_ids");
    for (int layer = 0; layer < 12; layer++) {
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".self_attn.q_proj.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".self_attn.k_proj.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".self_attn.v_proj.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".self_attn.out_proj.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".mlp.fc1.weight");
        keys.push_back("text_model.encoder.layers." + std::to_string(layer) + ".mlp.fc2.weight");
    }
    return keys;
}

// ---------------------------------------------------------------------------
// Helper: T5XXL safetensors keys
// ---------------------------------------------------------------------------
static std::vector<std::string> make_t5xxl_keys() {
    std::vector<std::string> keys;
    for (int block = 0; block < 24; block++) {
        keys.push_back("encoder.block." + std::to_string(block) + ".layer.0.SelfAttention.q.weight");
        keys.push_back("encoder.block." + std::to_string(block) + ".layer.0.SelfAttention.k.weight");
        keys.push_back("encoder.block." + std::to_string(block) + ".layer.0.SelfAttention.v.weight");
        keys.push_back("encoder.block." + std::to_string(block) + ".layer.0.SelfAttention.o.weight");
        keys.push_back("encoder.block." + std::to_string(block) + ".layer.1.DenseReluDense.wi_0.weight");
        keys.push_back("encoder.block." + std::to_string(block) + ".layer.1.DenseReluDense.wi_1.weight");
        keys.push_back("encoder.block." + std::to_string(block) + ".layer.1.DenseReluDense.wo.weight");
    }
    keys.push_back("encoder.final_layer_norm.weight");
    keys.push_back("shared.weight");
    return keys;
}

// ---------------------------------------------------------------------------
// Helper: VAE-like tensor keys (should default to "vae")
// ---------------------------------------------------------------------------
static std::vector<std::string> make_vae_keys() {
    std::vector<std::string> keys;
    keys.push_back("decoder.conv_in.weight");
    keys.push_back("decoder.conv_in.bias");
    keys.push_back("decoder.mid.block_1.conv1.weight");
    keys.push_back("encoder.conv_out.weight");
    return keys;
}

// =============================================================================
// Tests
// =============================================================================

static void test_gpt_oss_gguf() {
    auto keys = make_gpt_oss_gguf_keys();
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "llm", "GPT-OSS 20B GGUF should be detected as llm, got: " + result);
}

static void test_gpt_oss_safetensors() {
    auto keys = make_gpt_oss_safetensors_keys();
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "llm", "GPT-OSS 20B safetensors should be detected as llm, got: " + result);
}

static void test_standard_llm_gguf() {
    auto keys = make_standard_llm_gguf_keys();
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "llm", "Standard dense LLM (gguf) should be detected as llm, got: " + result);
}

static void test_clip_g() {
    auto keys = make_clip_g_keys();
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "clip_g", "CLIP-G should be detected as clip_g, got: " + result);
}

static void test_clip_l() {
    auto keys = make_clip_l_keys();
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "clip_l", "CLIP-L should be detected as clip_l, got: " + result);
}

static void test_t5xxl() {
    auto keys = make_t5xxl_keys();
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "t5xxl", "T5XXL should be detected as t5xxl, got: " + result);
}

static void test_vae() {
    auto keys = make_vae_keys();
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "vae", "VAE-like keys should default to vae, got: " + result);
}

static void test_empty_keys() {
    std::vector<std::string> keys;
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "vae", "Empty keys should default to vae, got: " + result);
}

// ---------------------------------------------------------------------------
// Edge case: GPT-OSS with minimal keys (just MoE MLP + attention, no token_embd)
// Should still detect as LLM via the (attention && mlp && output_norm) condition.
// ---------------------------------------------------------------------------
static void test_gpt_oss_minimal() {
    std::vector<std::string> keys;
    keys.push_back("blk.0.attn_q.weight");
    keys.push_back("blk.0.attn_output.weight");
    keys.push_back("blk.0.ffn_gate_inp.weight");
    keys.push_back("blk.0.ffn_gate_exps.0.weight");
    keys.push_back("output_norm.weight");
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "llm", "GPT-OSS minimal (MoE + attn + output_norm) should be detected as llm, got: " + result);
}

// ---------------------------------------------------------------------------
// Edge case: MoE keys using singular "exp" form (alternate GGUF variant)
// ---------------------------------------------------------------------------
static void test_moe_singular_exp_form() {
    std::vector<std::string> keys;
    keys.push_back("token_embd.weight");
    keys.push_back("blk.0.attn_q.weight");
    keys.push_back("blk.0.attn_k.weight");
    keys.push_back("blk.0.attn_v.weight");
    keys.push_back("blk.0.attn_output.weight");
    keys.push_back("blk.0.ffn_gate_inp.weight");
    keys.push_back("blk.0.ffn_gate_exp.0.weight");   // singular "exp"
    keys.push_back("blk.0.ffn_up_exp.0.weight");
    keys.push_back("blk.0.ffn_down_exp.0.weight");
    keys.push_back("output_norm.weight");
    std::string result = inferModelTypeFromTensorKeys(keys);
    ASSER_EQ(result, "llm", "MoE singular exp form should be detected as llm, got: " + result);
}

int main() {
    std::cout << "=== Model Detection Tests ===\n\n";

    test_gpt_oss_gguf();
    test_gpt_oss_safetensors();
    test_standard_llm_gguf();
    test_clip_g();
    test_clip_l();
    test_t5xxl();
    test_vae();
    test_empty_keys();
    test_gpt_oss_minimal();
    test_moe_singular_exp_form();

    std::cout << "\n=== Results: " << tests_passed << " passed, " 
              << tests_failed << " failed ===\n";
    return tests_failed > 0 ? 1 : 0;
}
