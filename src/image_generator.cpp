#include "image_generator.h"

#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "../stable-diffusion.cpp/thirdparty/stb_image.h"
#include "../stable-diffusion.cpp/thirdparty/stb_image_write.h"
#include "base64.hpp"
#include "crow.h"
#include "image_utils.h"
#include "logging.h"
#include "model_detection.h"
#include "server.h"
#include "string_utils.h"

namespace fs = std::filesystem;

// Global callback data structure
struct CallbackData {
    ImageGenerator* generator;
    std::string task_id;
    TaskStateManager* task_state_manager;
    int total_steps;
};

static CallbackData g_callback_data;
static std::mutex g_callback_mutex;

// Progress callback for stable-diffusion.cpp
static void progress_callback(int step, int steps, float time, void* data) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    if (g_callback_data.task_state_manager && !g_callback_data.task_id.empty()) {
        float progress = steps > 0 ? static_cast<float>(step) / static_cast<float>(steps) : 0.0f;
        g_callback_data.task_state_manager->updateTaskProgress(g_callback_data.task_id, progress);
        LOG_DEBUG("Progress: step %d/%d (%.1f%%), time: %.2fs", step, steps, progress * 100.0f, time);
    }
}

// Preview callback for stable-diffusion.cpp
static void preview_callback(int step, int frame_count, sd_image_t* frames, bool is_noisy, void* data) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    LOG_DEBUG("Preview callback: step %d, frame_count %d, is_noisy %d", step, frame_count, is_noisy);
    if (g_callback_data.task_state_manager && !g_callback_data.task_id.empty() && frames && frame_count > 0) {
        // Encode first frame to JPEG, then base64 for live preview
        std::vector<unsigned char> jpg_buffer;
        auto write_callback = [](void* context, void* data, int size) {
            auto* buffer = static_cast<std::vector<unsigned char>*>(context);
            unsigned char* bytes = static_cast<unsigned char*>(data);
            buffer->insert(buffer->end(), bytes, bytes + size);
        };

        // Use lower quality for faster preview encoding
        int quality = 75;
        int result = stbi_write_jpg_to_func(write_callback, &jpg_buffer, frames[0].width, frames[0].height,
                                            frames[0].channel, frames[0].data, quality);

        LOG_DEBUG("JPEG encoding result: %d, jpg_buffer size: %zu", result, jpg_buffer.size());
        if (result != 0 && !jpg_buffer.empty()) {
            std::string preview_base64 = base64_encode(jpg_buffer.data(), jpg_buffer.size());

            float progress = g_callback_data.total_steps > 0
                                 ? static_cast<float>(step) / static_cast<float>(g_callback_data.total_steps)
                                 : 0.0f;

            g_callback_data.task_state_manager->updateTaskProgress(g_callback_data.task_id, progress, preview_base64);

            LOG_DEBUG("Preview: step %d, frames: %d, noisy: %d, jpg_size: %zu bytes", step, frame_count, is_noisy,
                      jpg_buffer.size());
        } else {
            LOG_ERROR("Failed to encode preview image as JPEG");
        }
    }
}

ImageGenerator::ImageGenerator(std::shared_ptr<TaskStateManager> task_state_manager,
                               std::shared_ptr<OptionsManager> options_manager,
                               std::shared_ptr<ModelManager> model_manager, std::shared_ptr<ImageFilters> image_filters,
                               const ServerParams& server_params)
    : sd_ctx_(nullptr),
      task_state_manager_(task_state_manager),
      options_manager_(options_manager),
      model_manager_(model_manager),
      image_filters_(image_filters),
      initialized_(false),
      interrupted_(false),
      vae_on_cpu_(server_params.vae_on_cpu),
      vae_tiling_(server_params.vae_tiling),
      vae_tile_size_(server_params.vae_tile_size),
      offload_to_cpu_(server_params.offload_to_cpu),
      diffusion_fa_(server_params.diffusion_fa),
      control_net_cpu_(server_params.control_net_cpu),
      clip_on_cpu_(server_params.clip_on_cpu),
      chroma_disable_dit_mask_(server_params.chroma_disable_dit_mask) {
    LOG_INFO("ImageGenerator created");
}

ImageGenerator::~ImageGenerator() {
    if (sd_ctx_) {
        LOG_INFO("Freeing SD context");
        free_sd_ctx(sd_ctx_);
        sd_ctx_ = nullptr;
    }
}

// Helper function to build embedding map from directory
static void buildEmbeddingMap(const std::string& embedding_dir, std::map<std::string, std::string>& embedding_map) {
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt", ".sft"};

    if (embedding_dir.empty() || !fs::exists(embedding_dir) || !fs::is_directory(embedding_dir)) {
        return;
    }

    for (auto& p : fs::recursive_directory_iterator(embedding_dir)) {
        if (!p.is_regular_file()) continue;

        auto path = p.path();
        std::string ext = path_to_utf8(path.extension());

        bool valid = false;
        for (auto& e : valid_ext) {
            if (ext == e) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            continue;
        }

        std::string key = path_to_utf8(path.stem());
        std::string value = path_to_utf8(path);

        embedding_map[key] = value;
    }
}

bool ImageGenerator::isInitialized() const { return initialized_ && sd_ctx_ != nullptr; }

std::string ImageGenerator::getCurrentModelPath() const { return current_model_path_; }

void ImageGenerator::interrupt() {
    std::lock_guard<std::mutex> lock(mutex_);
    interrupted_ = true;
    LOG_INFO("Generation interrupted");
}

std::vector<std::string> ImageGenerator::generateTxt2Img(const ImageGenerationParams& params,
                                                         const std::string& task_id) {
    return generateInternal(params, false, task_id);
}

std::vector<std::string> ImageGenerator::generateImg2Img(const ImageGenerationParams& params,
                                                         const std::string& task_id) {
    return generateInternal(params, true, task_id);
}

std::vector<std::string> ImageGenerator::generateInternal(const ImageGenerationParams& params, bool is_img2img,
                                                          const std::string& task_id) {
    // Ensure the correct model is loaded based on current options and controlnet (before taking lock)
    if (!ensureModelLoaded(params.controlnet_model)) {
        LOG_ERROR("Failed to ensure model is loaded");
        throw std::runtime_error("Failed to load model from options");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    interrupted_ = false;
    current_task_id_ = task_id;

    // Set up callbacks
    {
        std::lock_guard<std::mutex> cb_lock(g_callback_mutex);
        g_callback_data.generator = this;
        g_callback_data.task_id = task_id;
        g_callback_data.task_state_manager = task_state_manager_.get();
        g_callback_data.total_steps = params.steps;
    }

    sd_set_progress_callback(progress_callback, nullptr);

    // Set preview callback only if live previews are enabled
    auto options = options_manager_->getOptions();
    std::string options_json = options.dump();
    auto parsed_options = crow::json::load(options_json);
    bool live_previews_enabled = true;  // default to true
    if (parsed_options && parsed_options.has("live_previews_enable")) {
        live_previews_enabled = parsed_options["live_previews_enable"].b();
    }

    if (live_previews_enabled) {
        sd_set_preview_callback(preview_callback, PREVIEW_PROJ, 3, true, false, nullptr);
    } else {
        // Clear any existing preview callback
        sd_set_preview_callback(nullptr, PREVIEW_PROJ, 3, true, false, nullptr);
    }

    LOG_INFO("Generating %s: prompt='%s', size=%dx%d, steps=%d, seed=%lld", is_img2img ? "img2img" : "txt2img",
             params.prompt.c_str(), params.width, params.height, params.steps, params.seed);

    std::vector<sd_lora_t> lora_vec;
    size_t lora_count = params.lora_paths.size();
    if (params.lora_alphas.size() < lora_count) {
        LOG_WARNING("LoRA alpha count (%zu) does not match LoRA path count (%zu)", params.lora_alphas.size(),
                    params.lora_paths.size());
        lora_count = params.lora_alphas.size();
    }

    for (size_t i = 0; i < lora_count; i++) {
        sd_lora_t item;
        item.is_high_noise = false;
        item.path = params.lora_paths[i].c_str();
        item.multiplier = params.lora_alphas[i];
        lora_vec.push_back(item);
    }

    if (!lora_vec.empty()) {
        LOG_INFO("Using %zu LoRA(s)", lora_vec.size());
        for (const auto& lora : lora_vec) {
            LOG_DEBUG("  LoRA: %s (%.2f)", lora.path, lora.multiplier);
        }
    }

    // Initialize generation parameters
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);

    // Set loras
    gen_params.loras = lora_vec.empty() ? nullptr : lora_vec.data();
    gen_params.lora_count = static_cast<uint32_t>(lora_vec.size());

    gen_params.prompt = params.prompt.c_str();
    gen_params.negative_prompt = params.negative_prompt.empty() ? "" : params.negative_prompt.c_str();
    gen_params.clip_skip = params.clip_skip;
    gen_params.width = params.width;
    gen_params.height = params.height;
    gen_params.batch_count = params.batch_count;
    gen_params.seed = params.seed < 0 ? static_cast<int64_t>(time(nullptr)) : params.seed;

    // Sample parameters
    gen_params.sample_params.sample_method = params.sampler;
    gen_params.sample_params.scheduler = params.scheduler;
    gen_params.sample_params.sample_steps = params.steps;
    gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;

    // img2img specific
    if (is_img2img && !params.init_image_base64.empty()) {
        sd_image_t init_image = createInitImage(params);
        sd_image_t mask_image = createMaskImage(params);

        gen_params.init_image = init_image;
        gen_params.mask_image = mask_image;
        gen_params.strength = params.strength;
    }

    // ControlNet specific
    sd_image_t control_image = {0, 0, 0, nullptr};
    if (!params.control_image_base64.empty() && !params.controlnet_model.empty()) {
        control_image = createControlImage(params);
        gen_params.control_image = control_image;
        gen_params.control_strength = params.control_strength;
        LOG_INFO("Using ControlNet with strength %.2f", params.control_strength);
    }

    // Reference images for vision-based models (Qwen, etc.)
    std::vector<sd_image_t> ref_images_vec = createRefImages(params);
    if (!ref_images_vec.empty()) {
        gen_params.ref_images = ref_images_vec.data();
        gen_params.ref_images_count = static_cast<int>(ref_images_vec.size());
        gen_params.auto_resize_ref_image = params.auto_resize_ref_image;
        LOG_INFO("Using %d reference image(s)", gen_params.ref_images_count);
    }

    // VAE tiling (if enabled via CLI)
    if (vae_tiling_) {
        gen_params.vae_tiling_params.enabled = true;

        // Parse tile size (in pixel space, will convert to latent space)
        int tile_size_x = 256;
        int tile_size_y = 256;
        if (!vae_tile_size_.empty()) {
            size_t x_pos = vae_tile_size_.find('x');
            try {
                if (x_pos != std::string::npos) {
                    std::string tile_x_str = vae_tile_size_.substr(0, x_pos);
                    std::string tile_y_str = vae_tile_size_.substr(x_pos + 1);
                    tile_size_x = std::stoi(tile_x_str);
                    tile_size_y = std::stoi(tile_y_str);
                } else {
                    tile_size_x = tile_size_y = std::stoi(vae_tile_size_);
                }
            } catch (const std::exception& e) {
                LOG_WARNING("Invalid VAE tile size '%s', using default 256x256", vae_tile_size_.c_str());
                tile_size_x = tile_size_y = 256;
            }
        }

        // Convert from pixel space to latent space (VAE downscaling factor is 8)
        int latent_tile_x = tile_size_x / 8;
        int latent_tile_y = tile_size_y / 8;

        gen_params.vae_tiling_params.tile_size_x = latent_tile_x;
        gen_params.vae_tiling_params.tile_size_y = latent_tile_y;
        gen_params.vae_tiling_params.target_overlap = 0.5f;
        gen_params.vae_tiling_params.rel_size_x = 0.0f;
        gen_params.vae_tiling_params.rel_size_y = 0.0f;
        LOG_INFO("VAE tiling enabled with tile size %dx%d pixels (%dx%d latent)", tile_size_x, tile_size_y,
                 latent_tile_x, latent_tile_y);
    }

    // Generate images
    sd_image_t* result = generate_image(sd_ctx_, &gen_params);

    // Free init image if used
    if (gen_params.init_image.data) {
        freeImage(gen_params.init_image);
    }
    if (gen_params.mask_image.data) {
        freeImage(gen_params.mask_image);
    }
    // Free control image if used
    if (control_image.data) {
        freeImage(control_image);
    }

    // Free reference images if used
    for (auto& ref_image : ref_images_vec) {
        if (ref_image.data) {
            freeImage(ref_image);
        }
    }

    if (!result) {
        LOG_ERROR("Image generation failed");

        // Clear callbacks
        {
            std::lock_guard<std::mutex> cb_lock(g_callback_mutex);
            g_callback_data.task_id.clear();
        }

        throw std::runtime_error("Image generation failed");
    }

    // Convert results to base64
    std::vector<std::string> result_images;
    for (int i = 0; i < params.batch_count; i++) {
        std::string img_base64 = imageToBase64(result[i]);
        result_images.push_back(img_base64);

        // Free the result image data
        if (result[i].data) {
            free(result[i].data);
        }
    }

    // Free result array
    free(result);

    // Clear callbacks
    {
        std::lock_guard<std::mutex> cb_lock(g_callback_mutex);
        g_callback_data.task_id.clear();
    }

    LOG_INFO("Generated %zu images successfully", result_images.size());

    return result_images;
}

sd_image_t ImageGenerator::createInitImage(const ImageGenerationParams& params) {
    sd_image_t init_image = base64ToImage(params.init_image_base64);
    if (!init_image.data) {
        throw std::runtime_error("Failed to decode init image");
    }

    if (!resizeImage(init_image, params.width, params.height, true)) {
        freeImage(init_image);
        throw std::runtime_error("Failed to resize init image");
    }

    return init_image;
}

sd_image_t ImageGenerator::createMaskImage(const ImageGenerationParams& params) {
    sd_image_t mask_image = {0, 0, 0, nullptr};

    if (params.mask_base64.empty()) {
        // Create a default mask for img2img (all white = no masking)
        mask_image.width = params.width;
        mask_image.height = params.height;
        mask_image.channel = 1;
        mask_image.data = static_cast<uint8_t*>(malloc(params.width * params.height * 1));
        if (!mask_image.data) {
            throw std::runtime_error("Failed to allocate mask image");
        }
        memset(mask_image.data, 255, params.width * params.height * 1);  // Fill with 255 (white)

        return mask_image;
    }

    // Decode provided mask
    mask_image = base64ToImage(params.mask_base64, 1);
    if (!mask_image.data) {
        throw std::runtime_error("Failed to decode mask image");
    }

    // Resize mask to match init image dimensions if needed
    if (mask_image.width != params.width || mask_image.height != params.height) {
        if (!resizeImage(mask_image, params.width, params.height, true)) {
            freeImage(mask_image);
            throw std::runtime_error("Failed to resize mask image");
        }
    }

    return mask_image;
}

sd_image_t ImageGenerator::createControlImage(const ImageGenerationParams& params) {
    sd_image_t control_image = {0, 0, 0, nullptr};

    if (params.control_image_base64.empty()) {
        return control_image;
    }

    // Decode control image from base64
    control_image = base64ToImage(params.control_image_base64);
    if (!control_image.data) {
        throw std::runtime_error("Failed to decode control image");
    }

    // Resize control image to match generation dimensions
    if (!resizeImage(control_image, params.width, params.height, true)) {
        freeImage(control_image);
        throw std::runtime_error("Failed to resize control image");
    }

    // Apply ControlNet preprocessing using ImageFilters
    sd_image_t processed_image = image_filters_->applyControlNetFilter(control_image, "canny");
    if (!processed_image.data) {
        LOG_WARNING("Failed to apply ControlNet preprocessing to control image, using original");
    } else {
        freeImage(control_image);
        control_image = processed_image;
        LOG_INFO("Applied ControlNet preprocessing to control image");
    }

    return control_image;
}

std::vector<sd_image_t> ImageGenerator::createRefImages(const ImageGenerationParams& params) {
    std::vector<sd_image_t> ref_images;

    if (params.ref_images_base64.empty()) {
        return ref_images;  // Return empty vector if no ref images
    }

    LOG_INFO("Processing %zu reference image(s)", params.ref_images_base64.size());

    for (size_t i = 0; i < params.ref_images_base64.size(); i++) {
        if (params.ref_images_base64[i].empty()) {
            LOG_WARNING("Skipping empty reference image at index %zu", i);
            continue;
        }

        // Decode reference image from base64
        sd_image_t ref_image = base64ToImage(params.ref_images_base64[i]);
        if (!ref_image.data) {
            LOG_ERROR("Failed to decode reference image at index %zu", i);
            // Free previously decoded images before throwing
            for (auto& img : ref_images) {
                freeImage(img);
            }
            throw std::runtime_error("Failed to decode reference image at index " + std::to_string(i));
        }

        // Resize reference image if auto_resize_ref_image is enabled
        if (params.auto_resize_ref_image) {
            if (!resizeImage(ref_image, params.width, params.height, true)) {
                LOG_ERROR("Failed to resize reference image at index %zu", i);
                freeImage(ref_image);
                // Free previously decoded images
                for (auto& img : ref_images) {
                    freeImage(img);
                }
                throw std::runtime_error("Failed to resize reference image at index " + std::to_string(i));
            }
        }

        LOG_DEBUG("Loaded reference image %zu: %ux%u channels", i, ref_image.width, ref_image.height,
                  ref_image.channel);
        ref_images.push_back(ref_image);
    }

    LOG_INFO("Successfully loaded %zu reference image(s)", ref_images.size());
    return ref_images;
}

void ImageGenerator::freeImage(sd_image_t& image) {
    if (image.data) {
        free(image.data);  // Works for both stbi allocated and manually allocated memory
        image.data = nullptr;
    }
    image.width = 0;
    image.height = 0;
    image.channel = 0;
}
bool ImageGenerator::needsModelReload(const std::string& model_path) const {
    // If not initialized, we need to load
    if (!initialized_ || !sd_ctx_) {
        return true;
    }

    // If model path changed, we need to reload
    if (model_path != current_model_path_) {
        return true;
    }

    return false;
}

bool ImageGenerator::ensureModelLoaded(const std::string& controlnet_model) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Get options
    auto options_wvalue = options_manager_->getOptions();
    std::string options_json = options_wvalue.dump();
    auto options = crow::json::load(options_json);

    if (!options) {
        LOG_ERROR("Failed to load options");
        return false;
    }

    // Get model path from options
    std::string model_path;
    if (options.has("sd_model_checkpoint")) {
        std::string model_name = std::string(options["sd_model_checkpoint"].s());

        if (!model_name.empty()) {
            // Get full path from model manager
            ModelInfo model_info = model_manager_->getModelByName(model_name, ModelType::CHECKPOINT);
            if (!model_info.full_path.empty()) {
                model_path = model_info.full_path;
            } else {
                LOG_ERROR("Model not found: %s", model_name.c_str());
                return false;
            }
        } else {
            LOG_ERROR("No model selected. Please configure sd_model_checkpoint in options.");
            return false;
        }
    } else {
        LOG_ERROR("No model selected. Please configure sd_model_checkpoint in options.");
        return false;
    }

    // Get ControlNet model path if specified
    std::string controlnet_path_str;
    if (!controlnet_model.empty()) {
        ModelInfo controlnet_info = model_manager_->getModelByName(controlnet_model, ModelType::CONTROLNET);
        if (!controlnet_info.full_path.empty()) {
            controlnet_path_str = controlnet_info.full_path;
            LOG_INFO("Found ControlNet model: %s -> %s", controlnet_model.c_str(), controlnet_path_str.c_str());
        } else {
            LOG_WARNING("ControlNet model not found: %s", controlnet_model.c_str());
        }
    }

    // Collect additional modules for comparison
    std::string vae_path_str;
    std::string clip_l_path_str;
    std::string clip_g_path_str;
    std::string t5xxl_path_str;
    std::string llm_path_str;

    if (options.has("forge_additional_modules")) {
        auto modules = options["forge_additional_modules"];
        if (modules.t() == crow::json::type::List) {
            for (size_t i = 0; i < modules.size(); i++) {
                std::string module_full_path = std::string(modules[i].s());
                if (module_full_path.empty()) {
                    continue;
                }

                LOG_DEBUG("Processing forge_additional_module: %s", module_full_path.c_str());

                // Inspect the model to determine its type
                std::string model_type = inspectModelType(module_full_path);

                if (model_type == "vae") {
                    vae_path_str = module_full_path;
                } else if (model_type == "clip_l") {
                    clip_l_path_str = module_full_path;
                } else if (model_type == "clip_g") {
                    clip_g_path_str = module_full_path;
                } else if (model_type == "t5xxl") {
                    t5xxl_path_str = module_full_path;
                } else if (model_type == "llm") {
                    llm_path_str = module_full_path;
                } else {
                    LOG_WARNING("Unknown model type for: %s (detected as: %s)", module_full_path.c_str(),
                                model_type.c_str());
                }
            }
        }
    }

    std::string lora_dir_str = model_manager_->getLoraDir();
    std::string embeddings_dir_str = model_manager_->getEmbeddingsDir();

    // Check if we need to reload the model (check all paths including controlnet)
    bool needs_reload = !initialized_ || !sd_ctx_ || model_path != current_model_path_ ||
                        vae_path_str != current_vae_path_ || clip_l_path_str != current_clip_l_path_ ||
                        clip_g_path_str != current_clip_g_path_ || t5xxl_path_str != current_t5xxl_path_ ||
                        llm_path_str != current_llm_path_ || controlnet_path_str != current_controlnet_path_ ||
                        embeddings_dir_str != current_embeddings_dir_;

    if (!needs_reload) {
        LOG_DEBUG("Model already loaded: %s", model_path.c_str());
        return true;
    }

    LOG_INFO("Model change detected, loading new model: %s", model_path.c_str());

    // Free old context if exists
    if (sd_ctx_) {
        LOG_INFO("Freeing old SD context for model switch");
        free_sd_ctx(sd_ctx_);
        sd_ctx_ = nullptr;
        initialized_ = false;
    }

    // Initialize with new model (mutex already held, don't call initialize which would deadlock)
    sd_set_log_callback(sd_log_cb, nullptr);

    if (model_path.empty()) {
        LOG_ERROR("Model path is empty");
        return false;
    }

    LOG_INFO("Initializing SD context with model: %s", model_path.c_str());

    // Initialize context parameters
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);

    params.free_params_immediately = false;

    // Check if we have additional modules (VAE, CLIP, etc.)
    bool has_additional_modules =
        !clip_l_path_str.empty() || !clip_g_path_str.empty() || !t5xxl_path_str.empty() || !llm_path_str.empty();

    if (has_additional_modules) {
        LOG_INFO("Using additional modules - loading diffusion model separately");
        // When using additional modules, load diffusion model separately
        params.model_path = nullptr;
        params.diffusion_model_path = model_path.c_str();
    } else {
        // Standard loading from checkpoint
        params.model_path = model_path.c_str();
        params.diffusion_model_path = nullptr;
    }

    params.vae_path = vae_path_str.empty() ? nullptr : vae_path_str.c_str();
    params.clip_l_path = clip_l_path_str.empty() ? nullptr : clip_l_path_str.c_str();
    params.clip_g_path = clip_g_path_str.empty() ? nullptr : clip_g_path_str.c_str();
    params.t5xxl_path = t5xxl_path_str.empty() ? nullptr : t5xxl_path_str.c_str();
    params.llm_path = llm_path_str.empty() ? nullptr : llm_path_str.c_str();
    params.taesd_path = nullptr;
    params.control_net_path = controlnet_path_str.empty() ? nullptr : controlnet_path_str.c_str();

    // Build embeddings
    std::map<std::string, std::string> embedding_map;
    std::vector<sd_embedding_t> embedding_vec;
    buildEmbeddingMap(embeddings_dir_str, embedding_map);

    if (!embedding_map.empty()) {
        embedding_vec.reserve(embedding_map.size());
        for (const auto& kv : embedding_map) {
            sd_embedding_t item;
            item.name = kv.first.c_str();
            item.path = kv.second.c_str();
            embedding_vec.emplace_back(item);
        }
        params.embeddings = embedding_vec.data();
        params.embedding_count = static_cast<uint32_t>(embedding_vec.size());
        LOG_INFO("Loading %zu embedding(s)", embedding_vec.size());
    } else {
        params.embeddings = nullptr;
        params.embedding_count = 0;
    }

    params.vae_decode_only = false;  // We need encoding for img2img

    // Log what we're loading
    if (!vae_path_str.empty()) {
        LOG_INFO("Loading VAE model: %s", vae_path_str.c_str());
    }
    if (!clip_l_path_str.empty()) {
        LOG_INFO("Loading CLIP-L model: %s", clip_l_path_str.c_str());
    }
    if (!clip_g_path_str.empty()) {
        LOG_INFO("Loading CLIP-G model: %s", clip_g_path_str.c_str());
    }
    if (!t5xxl_path_str.empty()) {
        LOG_INFO("Loading T5XXL model: %s", t5xxl_path_str.c_str());
    }
    if (!llm_path_str.empty()) {
        LOG_INFO("Loading LLM model: %s", llm_path_str.c_str());
    }
    if (!controlnet_path_str.empty()) {
        LOG_INFO("Loading ControlNet model: %s", controlnet_path_str.c_str());
    }
    if (!lora_dir_str.empty()) {
        LOG_DEBUG("LoRA model directory: %s", lora_dir_str.c_str());
    }
    if (!embeddings_dir_str.empty()) {
        LOG_DEBUG("Embeddings directory: %s", embeddings_dir_str.c_str());
    }

    // Set RNG type
    params.rng_type = CUDA_RNG;

    // Apply CLI parameters for SD context
    params.keep_vae_on_cpu = vae_on_cpu_;
    params.offload_params_to_cpu = offload_to_cpu_;
    params.diffusion_flash_attn = diffusion_fa_;
    params.keep_control_net_on_cpu = control_net_cpu_;
    params.keep_clip_on_cpu = clip_on_cpu_;
    params.chroma_use_dit_mask = !chroma_disable_dit_mask_;

    if (vae_on_cpu_) {
        LOG_INFO("VAE will be kept on CPU");
    }
    if (offload_to_cpu_) {
        LOG_INFO("Parameters will be offloaded to CPU");
    }
    if (diffusion_fa_) {
        LOG_INFO("Diffusion flash attention enabled");
    }
    if (control_net_cpu_) {
        LOG_INFO("ControlNet will be kept on CPU");
    }
    if (clip_on_cpu_) {
        LOG_INFO("CLIP will be kept on CPU");
    }
    if (chroma_disable_dit_mask_) {
        LOG_INFO("DiT mask disabled for Chroma models");
    }

    // Create SD context
    sd_ctx_ = new_sd_ctx(&params);
    if (!sd_ctx_) {
        LOG_ERROR("Failed to create SD context");
        return false;
    }

    // Track currently loaded model paths
    current_model_path_ = model_path;
    current_vae_path_ = vae_path_str;
    current_clip_l_path_ = clip_l_path_str;
    current_clip_g_path_ = clip_g_path_str;
    current_t5xxl_path_ = t5xxl_path_str;
    current_llm_path_ = llm_path_str;
    current_taesd_path_ = "";
    current_lora_model_dir_ = lora_dir_str;
    current_embeddings_dir_ = embeddings_dir_str;
    current_controlnet_path_ = controlnet_path_str;

    initialized_ = true;
    LOG_INFO("SD context initialized successfully");

    return true;
}
