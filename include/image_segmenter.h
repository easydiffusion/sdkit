#ifndef __IMAGE_SEGMENTER_H__
#define __IMAGE_SEGMENTER_H__

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "model_manager.h"
#include "sam3.h"

struct SegmentationParams {
    std::string image_base64;  // Input image (PNG/JPEG/etc), base64-encoded (data URI prefix allowed)
    std::string model_name;    // Name of the segmentation model file in the segmentation models directory

    // Prompting. At least one must be provided.
    std::string text_prompt;   // Text-prompted detection (full SAM3 models only)
    std::vector<std::pair<float, float>> positive_points;  // (x, y) coordinates
    std::vector<std::pair<float, float>> negative_points;  // (x, y) coordinates
    std::vector<std::vector<float>> boxes;                 // [x0, y0, x1, y1] each

    float score_threshold = 0.5f;  // For text-prompted detection
};

struct SegmentationResult {
    std::vector<std::string> masks;      // Binary masks as base64-encoded grayscale PNGs
    std::vector<float> scores;           // Detection score per mask
};

class ImageSegmenter {
   public:
    explicit ImageSegmenter(std::shared_ptr<ModelManager> model_manager);
    ~ImageSegmenter();

    // Runs segmentation on the given image. Throws std::runtime_error on failure.
    SegmentationResult segment(const SegmentationParams& params);

   private:
    // Lazily loads the requested model, unloading a previously loaded one if needed.
    void ensureModelLoaded(const std::string& model_name);

    std::shared_ptr<ModelManager> model_manager_;

    // Guards all sam3 state; sam3 has no thread-safety guarantees of its own.
    std::mutex mutex_;

    std::string loaded_model_name_;
    std::shared_ptr<sam3_model> model_;
    sam3_state_ptr state_;
    sam3_params load_params_;
};

#endif  // __IMAGE_SEGMENTER_H__
