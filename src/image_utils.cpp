#include "image_utils.h"

#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE2_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize2.h"
#include "base64.hpp"
#include "logging.h"

std::string imageToBase64(const sd_image_t& image) {
    if (!image.data || image.width == 0 || image.height == 0) {
        return "";
    }

    // Use stb_image_write to convert to PNG format
    std::vector<unsigned char> png_buffer;

    // Lambda function to capture PNG data
    auto write_callback = [](void* context, void* data, int size) {
        auto* buffer = static_cast<std::vector<unsigned char>*>(context);
        unsigned char* bytes = static_cast<unsigned char*>(data);
        buffer->insert(buffer->end(), bytes, bytes + size);
    };

    // Write PNG to buffer (stride is width * channels for tightly packed data)
    int stride = image.width * image.channel;
    int result = stbi_write_png_to_func(write_callback, &png_buffer, image.width, image.height, image.channel,
                                        image.data, stride);

    if (result == 0 || png_buffer.empty()) {
        LOG_ERROR("Failed to encode image as PNG");
        return "";
    }

    // Encode PNG data as base64
    return base64_encode(png_buffer.data(), png_buffer.size());
}

sd_image_t base64ToImage(const std::string& base64_data, int desired_channels) {
    sd_image_t image = {0, 0, 0, nullptr};

    if (base64_data.empty()) {
        return image;
    }

    // Strip data URI prefix if present (e.g., "data:image/png;base64,")
    std::string base64_clean = base64_data;
    size_t comma_pos = base64_data.find(',');
    if (comma_pos != std::string::npos) {
        // Check if this looks like a data URI
        if (base64_data.substr(0, 5) == "data:") {
            base64_clean = base64_data.substr(comma_pos + 1);
            LOG_DEBUG("Stripped data URI prefix, base64 length: %zu", base64_clean.length());
        }
    }

    // Decode base64
    std::vector<unsigned char> decoded_data = base64_decode(base64_clean);

    if (decoded_data.empty()) {
        LOG_ERROR("Failed to decode base64 data");
        return image;
    }

    // Load image using stb_image
    int width, height, channels;
    unsigned char* img_data =
        stbi_load_from_memory(decoded_data.data(), decoded_data.size(), &width, &height, &channels, desired_channels);

    if (!img_data) {
        LOG_ERROR("Failed to load image from memory: %s", stbi_failure_reason());
        return image;
    }

    // Use the desired channels if specified, otherwise use the actual channels
    int final_channels = desired_channels > 0 ? desired_channels : channels;

    image.width = width;
    image.height = height;
    image.channel = final_channels;
    image.data = img_data;

    LOG_DEBUG("Decoded image: %dx%d, %d channels", width, height, final_channels);

    return image;
}

void freeImage(sd_image_t& image) {
    if (image.data) {
        stbi_image_free(image.data);
        image.data = nullptr;
    }
    image.width = 0;
    image.height = 0;
    image.channel = 0;
}

bool resizeImage(sd_image_t& image, int target_width, int target_height, bool clamp_to_8) {
    if (!image.data || image.width == 0 || image.height == 0) {
        LOG_ERROR("Cannot resize invalid image");
        return false;
    }

    if (target_width <= 0 || target_height <= 0) {
        LOG_ERROR("Invalid target dimensions: %dx%d", target_width, target_height);
        return false;
    }

    // Clamp to multiples of 8 if requested
    int final_width = target_width;
    int final_height = target_height;
    if (clamp_to_8) {
        final_width = target_width - (target_width % 8);
        final_height = target_height - (target_height % 8);
        if (final_width <= 0) final_width = 8;
        if (final_height <= 0) final_height = 8;
    }

    if (image.width == final_width && image.height == final_height) {
        // Already at target size
        return true;
    }

    // Resize using stb_image_resize2
    // Map the channel count to stb's pixel layout enum (STBIR_1CH=0 .. STBIR_4CH=3)
    if (image.channel < 1 || image.channel > 4) {
        LOG_ERROR("Unsupported channel count for resize: %d", image.channel);
        return false;
    }
    stbir_pixel_layout pixel_layout = static_cast<stbir_pixel_layout>(image.channel - 1);

    // Allocate new buffer for resized image
    int new_size = final_width * final_height * image.channel;
    unsigned char* resized_data = (unsigned char*)malloc(new_size);
    if (!resized_data) {
        LOG_ERROR("Failed to allocate memory for resized image");
        return false;
    }

    // Returns the output buffer on success, NULL on failure
    unsigned char* result = stbir_resize_uint8_linear(image.data, image.width, image.height, 0, resized_data,
                                                      final_width, final_height, 0, pixel_layout);

    if (result == nullptr) {
        LOG_ERROR("Failed to resize image");
        free(resized_data);
        return false;
    }

    // Free old data and update image
    stbi_image_free(image.data);
    image.data = resized_data;
    image.width = final_width;
    image.height = final_height;

    LOG_DEBUG("Resized image to %dx%d", final_width, final_height);
    return true;
}
