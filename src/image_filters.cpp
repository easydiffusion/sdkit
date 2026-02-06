#include "image_filters.h"

#include <filesystem>
#include <stdexcept>

#include "image_utils.h"
#include "logging.h"

namespace fs = std::filesystem;

ImageFilters::ImageFilters(std::shared_ptr<ModelManager> model_manager)
    : model_manager_(model_manager), upscaler_ctx_(nullptr) {
    LOG_DEBUG("ImageFilters created");
}

ImageFilters::~ImageFilters() {
    if (upscaler_ctx_) {
        LOG_INFO("Freeing upscaler context");
        free_upscaler_ctx(upscaler_ctx_);
        upscaler_ctx_ = nullptr;
    }
}

std::vector<std::string> ImageFilters::upscaleBatch(const std::vector<std::string>& base64_images,
                                                    const std::string& upscaler_name, int upscale_factor) {
    std::vector<std::string> result_images;
    if (!ensureUpscalerLoaded(upscaler_name)) {
        LOG_ERROR("Upscaler not available. Cannot upscale images.");
        return result_images;
    }

    for (size_t i = 0; i < base64_images.size(); i++) {
        // Convert base64 to sd_image_t
        sd_image_t input_image = base64ToImage(base64_images[i]);
        if (!input_image.data) {
            LOG_ERROR("Failed to decode image %zu", i);
            result_images.push_back("");
            continue;
        }

        // Upscale the image
        sd_image_t upscaled_image = upscaleImage(input_image, upscale_factor);

        // Free input image
        freeImage(input_image);

        if (!upscaled_image.data) {
            LOG_ERROR("Failed to upscale image %zu", i);
            result_images.push_back("");
            continue;
        }

        // Convert upscaled image back to base64
        std::string upscaled_base64 = imageToBase64(upscaled_image);
        result_images.push_back(upscaled_base64);

        // Free upscaled image
        freeImage(upscaled_image);

        LOG_INFO("Upscaled image %zu: %dx%d -> %dx%d", i, input_image.width, input_image.height, upscaled_image.width,
                 upscaled_image.height);
    }

    return result_images;
}

sd_image_t ImageFilters::upscaleImage(const sd_image_t& input_image, int upscale_factor) {
    sd_image_t upscaled_image = {0, 0, 0, nullptr};

    if (!input_image.data) {
        LOG_ERROR("Cannot upscale invalid image");
        return upscaled_image;
    }

    if (!upscaler_ctx_) {
        LOG_ERROR("Upscaler not loaded");
        return upscaled_image;
    }

    LOG_DEBUG("Upscaling image from %dx%d, factor %d", input_image.width, input_image.height, upscale_factor);
    upscaled_image = upscale(upscaler_ctx_, input_image, upscale_factor);

    if (upscaled_image.data) {
        LOG_DEBUG("Upscaled to %dx%d", upscaled_image.width, upscaled_image.height);
    }

    return upscaled_image;
}

std::vector<std::string> ImageFilters::applyControlNetFilterBatch(const std::vector<std::string>& base64_images,
                                                                  const std::string& module) {
    std::vector<std::string> result_images;

    for (size_t i = 0; i < base64_images.size(); i++) {
        // Convert base64 to sd_image_t
        sd_image_t input_image = base64ToImage(base64_images[i]);
        if (!input_image.data) {
            LOG_ERROR("Failed to decode image %zu for ControlNet processing", i);
            result_images.push_back("");
            continue;
        }

        // Apply ControlNet preprocessing using the single-image function
        sd_image_t processed_image = applyControlNetFilter(input_image, module);

        // Free input image
        freeImage(input_image);

        if (!processed_image.data) {
            LOG_ERROR("Failed to apply ControlNet preprocessing to image %zu", i);
            result_images.push_back("");
            continue;
        }

        // Convert processed image back to base64
        std::string processed_base64 = imageToBase64(processed_image);
        result_images.push_back(processed_base64);

        // Free processed image
        freeImage(processed_image);
    }

    return result_images;
}

sd_image_t ImageFilters::applyControlNetFilter(const sd_image_t& input_image, const std::string& module) {
    sd_image_t result_image = {0, 0, 0, nullptr};

    if (!input_image.data) {
        LOG_ERROR("Cannot apply ControlNet filter to invalid image");
        return result_image;
    }

    // Copy the input image data
    result_image.width = input_image.width;
    result_image.height = input_image.height;
    result_image.channel = input_image.channel;
    size_t data_size = input_image.width * input_image.height * input_image.channel * sizeof(uint8_t);
    result_image.data = (uint8_t*)malloc(data_size);
    if (!result_image.data) {
        LOG_ERROR("Failed to allocate memory for ControlNet filtered image");
        return result_image;
    }
    memcpy(result_image.data, input_image.data, data_size);

    // Apply ControlNet preprocessing (for now, always use canny)
    bool preprocess_result = preprocess_canny(result_image, 0.08f, 0.08f, 0.8f, 1.0f, false);
    if (!preprocess_result) {
        LOG_WARNING("Failed to apply ControlNet preprocessing (%s), using original", module.c_str());
    } else {
        LOG_INFO("Applied ControlNet preprocessing (%s)", module.c_str());
    }

    return result_image;
}

bool ImageFilters::ensureUpscalerLoaded(const std::string& upscaler_name) {
    // Upscaler name is required
    if (upscaler_name.empty()) {
        LOG_ERROR("No upscaler model name specified");
        return false;
    }

    // If upscaler is already loaded with the same name, we're good
    if (upscaler_ctx_ && current_upscaler_path_ == upscaler_name) {
        return true;
    }

    // Free old upscaler if exists
    if (upscaler_ctx_) {
        LOG_INFO("Freeing old upscaler context");
        free_upscaler_ctx(upscaler_ctx_);
        upscaler_ctx_ = nullptr;
        current_upscaler_path_.clear();
    }

    // Get the realesrgan models directory
    std::string realesrgan_dir = model_manager_->getRealesrganModelsPath();
    if (realesrgan_dir.empty()) {
        throw std::runtime_error("RealESRGAN models directory not set. Use --realesrgan-models-path argument.");
    }

    // Construct full path to the model
    fs::path model_path = fs::path(realesrgan_dir) / (upscaler_name + ".pth");
    std::string upscaler_path = model_path.string();

    LOG_INFO("Loading upscaler from: %s", upscaler_path.c_str());

    upscaler_ctx_ = new_upscaler_ctx(upscaler_path.c_str(), false, false, -1, 128);

    if (!upscaler_ctx_) {
        LOG_ERROR("Failed to load upscaler from: %s", upscaler_path.c_str());
        return false;
    }

    current_upscaler_path_ = upscaler_name;
    LOG_INFO("Upscaler loaded successfully");

    return true;
}