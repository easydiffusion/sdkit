#ifndef __FILTERS_CONTROLNET_HPP__
#define __FILTERS_CONTROLNET_HPP__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

#include "stable-diffusion.h"

namespace controlnet {

struct CannyOptions {
  float high_threshold = 0.08f;
  float low_threshold = 0.08f;
  float weak = 0.8f;
  float strong = 1.0f;
  bool inverse = false;
};

namespace detail {

constexpr float kPi = 3.14159265358979323846f;

struct FloatImage {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<float> data;

  FloatImage() = default;

  FloatImage(int w, int h, int c)
      : width(w), height(h), channels(c),
        data(static_cast<size_t>(w) * static_cast<size_t>(h) *
                 static_cast<size_t>(c),
             0.0f) {}

  bool empty() const {
    return width <= 0 || height <= 0 || channels <= 0 || data.empty();
  }

  size_t offset(int x, int y, int c = 0) const {
    return (static_cast<size_t>(y) * static_cast<size_t>(width) +
            static_cast<size_t>(x)) *
               static_cast<size_t>(channels) +
           static_cast<size_t>(c);
  }

  float &at(int x, int y, int c = 0) { return data[offset(x, y, c)]; }

  float at(int x, int y, int c = 0) const { return data[offset(x, y, c)]; }
};

inline char asciiToLower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

inline bool equalsIgnoreCase(std::string_view lhs, std::string_view rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (asciiToLower(lhs[i]) != asciiToLower(rhs[i])) {
      return false;
    }
  }

  return true;
}

inline uint8_t floatToU8(float value) {
  if (value <= 0.0f) {
    return 0;
  }
  if (value >= 1.0f) {
    return 255;
  }
  return static_cast<uint8_t>(value * 255.0f + 0.5f);
}

inline FloatImage sdImageToFloatImage(sd_image_t image) {
  FloatImage out(static_cast<int>(image.width), static_cast<int>(image.height),
                 static_cast<int>(image.channel));

  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      for (int c = 0; c < out.channels; ++c) {
        size_t index = out.offset(x, y, c);
        out.data[index] = static_cast<float>(image.data[index]) / 255.0f;
      }
    }
  }

  return out;
}

inline void floatImageToSdImage(const FloatImage &image, uint8_t *image_data) {
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      for (int c = 0; c < image.channels; ++c) {
        size_t index = image.offset(x, y, c);
        image_data[index] = floatToU8(image.data[index]);
      }
    }
  }
}

inline FloatImage gaussianKernel(int kernel_size) {
  FloatImage kernel(kernel_size, kernel_size, 1);
  int ks_mid = kernel_size / 2;
  float sigma = 1.4f;
  float normal = 1.0f / (2.0f * kPi * std::pow(sigma, 2.0f));

  for (int y = 0; y < kernel_size; ++y) {
    float gx = static_cast<float>(-ks_mid + y);
    for (int x = 0; x < kernel_size; ++x) {
      float gy = static_cast<float>(-ks_mid + x);
      float value =
          std::exp(-((gx * gx + gy * gy) / (2.0f * std::pow(sigma, 2.0f)))) *
          normal;
      kernel.at(x, y, 0) = value;
    }
  }

  return kernel;
}

inline FloatImage convolve(const FloatImage &input, const FloatImage &kernel,
                           int padding) {
  FloatImage output(input.width, input.height, input.channels);

  for (int c = 0; c < input.channels; ++c) {
    for (int y = 0; y < input.height; ++y) {
      for (int x = 0; x < input.width; ++x) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel.height; ++ky) {
          int iy = y + ky - padding;
          if (iy < 0 || iy >= input.height) {
            continue;
          }
          for (int kx = 0; kx < kernel.width; ++kx) {
            int ix = x + kx - padding;
            if (ix < 0 || ix >= input.width) {
              continue;
            }
            sum += input.at(ix, iy, c) * kernel.at(kx, ky, 0);
          }
        }
        output.at(x, y, c) = sum;
      }
    }
  }

  return output;
}

inline FloatImage grayscale(const FloatImage &rgb_image) {
  FloatImage grayscale_image(rgb_image.width, rgb_image.height, 1);

  for (int y = 0; y < rgb_image.height; ++y) {
    for (int x = 0; x < rgb_image.width; ++x) {
      float gray = 0.0f;
      if (rgb_image.channels == 1) {
        gray = rgb_image.at(x, y, 0);
      } else {
        float r = rgb_image.at(x, y, 0);
        float g = rgb_image.channels > 1 ? rgb_image.at(x, y, 1) : r;
        float b = rgb_image.channels > 2 ? rgb_image.at(x, y, 2) : g;
        gray = 0.2989f * r + 0.5870f * g + 0.1140f * b;
      }
      grayscale_image.at(x, y, 0) = gray;
    }
  }

  return grayscale_image;
}

inline FloatImage tensorHypot(const FloatImage &x, const FloatImage &y) {
  FloatImage out(x.width, x.height, x.channels);
  for (size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = std::sqrt(x.data[i] * x.data[i] + y.data[i] * y.data[i]);
  }
  return out;
}

inline FloatImage tensorArctan2(const FloatImage &x, const FloatImage &y) {
  FloatImage out(x.width, x.height, x.channels);
  for (size_t i = 0; i < out.data.size(); ++i) {
    out.data[i] = std::atan2(y.data[i], x.data[i]);
  }
  return out;
}

inline void normalize(FloatImage *image) {
  if (image == nullptr || image->empty()) {
    return;
  }

  float max_value = -std::numeric_limits<float>::infinity();
  for (float value : image->data) {
    if (value > max_value) {
      max_value = value;
    }
  }

  if (max_value == 0.0f || !std::isfinite(max_value)) {
    return;
  }

  float scale = 1.0f / max_value;
  for (float &value : image->data) {
    value *= scale;
  }
}

inline FloatImage nonMaxSuppression(const FloatImage &gradients,
                                    const FloatImage &directions) {
  FloatImage result(gradients.width, gradients.height, gradients.channels);

  // Keep the branch logic aligned with stable-diffusion.cpp's current
  // preprocess_canny() so the local v3 implementation matches the existing
  // canny behavior.
  for (int y = 1; y < result.height - 1; ++y) {
    for (int x = 1; x < result.width - 1; ++x) {
      float angle = directions.at(x, y, 0) * 180.0f / kPi;
      angle = angle < 0.0f ? angle + 180.0f : angle;
      float q = 1.0f;
      float r = 1.0f;

      if ((0.0f >= angle && angle < 22.5f) ||
          (157.5f >= angle && angle <= 180.0f)) {
        q = gradients.at(x, y + 1, 0);
        r = gradients.at(x, y - 1, 0);
      } else if (22.5f >= angle && angle < 67.5f) {
        q = gradients.at(x + 1, y - 1, 0);
        r = gradients.at(x - 1, y + 1, 0);
      } else if (67.5f >= angle && angle < 112.5f) {
        q = gradients.at(x + 1, y, 0);
        r = gradients.at(x - 1, y, 0);
      } else if (112.5f >= angle && angle < 157.5f) {
        q = gradients.at(x - 1, y - 1, 0);
        r = gradients.at(x + 1, y + 1, 0);
      }

      float current = gradients.at(x, y, 0);
      result.at(x, y, 0) = (current >= q && current >= r) ? current : 0.0f;
    }
  }

  return result;
}

inline void thresholdHysteresis(FloatImage *image, float high_threshold,
                                float low_threshold, float weak, float strong) {
  if (image == nullptr || image->empty()) {
    return;
  }

  float max_value = -std::numeric_limits<float>::infinity();
  for (float value : image->data) {
    if (value > max_value) {
      max_value = value;
    }
  }

  float high = max_value * high_threshold;
  float low = high * low_threshold;

  for (float &value : image->data) {
    if (value >= high) {
      value = strong;
    } else if (value <= high && value >= low) {
      value = weak;
    }
  }

  for (int y = 0; y < image->height; ++y) {
    for (int x = 0; x < image->width; ++x) {
      if (!(x >= 3 && x <= image->width - 3 && y >= 3 &&
            y <= image->height - 3)) {
        image->at(x, y, 0) = 0.0f;
      }
    }
  }

  // Keep the neighbor checks aligned with stable-diffusion.cpp's current
  // preprocess_canny() so the local v3 implementation matches the existing
  // canny behavior.
  for (int y = 1; y < image->height - 1; ++y) {
    for (int x = 1; x < image->width - 1; ++x) {
      float value = image->at(x, y, 0);
      if (value == weak) {
        bool has_strong_neighbor = image->at(x + 1, y - 1, 0) == strong ||
                                   image->at(x + 1, y, 0) == strong ||
                                   image->at(x, y - 1, 0) == strong ||
                                   image->at(x, y + 1, 0) == strong ||
                                   image->at(x - 1, y - 1, 0) == strong ||
                                   image->at(x - 1, y, 0) == strong;
        image->at(x, y, 0) = has_strong_neighbor ? strong : 0.0f;
      }
    }
  }
}

} // namespace detail

inline bool preprocess_canny(sd_image_t image,
                             const CannyOptions &options = {}) {
  if (image.data == nullptr || image.width == 0 || image.height == 0 ||
      image.channel == 0) {
    return false;
  }

  static constexpr float kSobelX[9] = {
      -1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f,
  };

  static constexpr float kSobelY[9] = {
      1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -1.0f,
  };

  detail::FloatImage gaussian = detail::gaussianKernel(5);
  detail::FloatImage sobel_x(3, 3, 1);
  detail::FloatImage sobel_y(3, 3, 1);
  sobel_x.data.assign(kSobelX, kSobelX + 9);
  sobel_y.data.assign(kSobelY, kSobelY + 9);

  detail::FloatImage image_f = detail::sdImageToFloatImage(image);
  detail::FloatImage grayscale = detail::grayscale(image_f);
  grayscale = detail::convolve(grayscale, gaussian, 2);
  detail::FloatImage grad_x = detail::convolve(grayscale, sobel_x, 1);
  detail::FloatImage grad_y = detail::convolve(grayscale, sobel_y, 1);
  detail::FloatImage magnitude = detail::tensorHypot(grad_x, grad_y);
  detail::normalize(&magnitude);
  detail::FloatImage theta = detail::tensorArctan2(grad_x, grad_y);
  grayscale = detail::nonMaxSuppression(magnitude, theta);
  detail::thresholdHysteresis(&grayscale, options.high_threshold,
                              options.low_threshold, options.weak,
                              options.strong);

  for (int y = 0; y < image_f.height; ++y) {
    for (int x = 0; x < image_f.width; ++x) {
      float gray = grayscale.at(x, y, 0);
      gray = options.inverse ? 1.0f - gray : gray;
      for (int c = 0; c < image_f.channels; ++c) {
        image_f.at(x, y, c) = gray;
      }
    }
  }

  detail::floatImageToSdImage(image_f, image.data);
  return true;
}

inline bool preprocess_canny(sd_image_t image, float high_threshold,
                             float low_threshold, float weak, float strong,
                             bool inverse) {
  return preprocess_canny(image, CannyOptions{high_threshold, low_threshold,
                                              weak, strong, inverse});
}

inline bool preprocess(sd_image_t image, std::string_view module) {
  if (module.empty() || detail::equalsIgnoreCase(module, "canny")) {
    return preprocess_canny(image);
  }

  return false;
}

} // namespace controlnet

#endif // __FILTERS_CONTROLNET_HPP__
