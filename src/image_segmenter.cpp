#include "image_segmenter.h"

#include <stdexcept>
#include <thread>

#include "image_utils.h"
#include "logging.h"

ImageSegmenter::ImageSegmenter(std::shared_ptr<ModelManager> model_manager) : model_manager_(model_manager) {}

ImageSegmenter::~ImageSegmenter() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_.reset();
    if (model_) {
        sam3_free_model(*model_);
        model_.reset();
    }
}

void ImageSegmenter::ensureModelLoaded(const std::string& model_name) {
    if (!model_name.empty() && model_name == loaded_model_name_ && model_) {
        return;
    }

    if (model_name.empty()) {
        throw std::runtime_error("No segmentation model specified");
    }

    if (model_manager_->getSegmentationDir().empty()) {
        throw std::runtime_error(
            "Segmentation is not available. Please start the server with --segmentation-models-path to enable it");
    }

    ModelInfo info = model_manager_->getModelByName(model_name, ModelType::SEGMENTATION);
    if (info.full_path.empty()) {
        throw std::runtime_error("Segmentation model not found: " + model_name);
    }

    LOG_INFO("Loading segmentation model: %s", info.full_path.c_str());

    // Release the old state and model before loading the new one
    state_.reset();
    if (model_) {
        sam3_free_model(*model_);
        model_.reset();
        loaded_model_name_.clear();
    }

    load_params_ = sam3_params{};
    load_params_.model_path = info.full_path;
    load_params_.use_gpu = true;  // Falls back to CPU when no GPU backend is available
    load_params_.n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (load_params_.n_threads <= 0) {
        load_params_.n_threads = 4;
    }

    model_ = sam3_load_model(load_params_);
    if (!model_) {
        throw std::runtime_error("Failed to load segmentation model: " + info.full_path);
    }

    state_ = sam3_create_state(*model_, load_params_);
    if (!state_) {
        sam3_free_model(*model_);
        model_.reset();
        throw std::runtime_error("Failed to create inference state for segmentation model: " + info.full_path);
    }

    loaded_model_name_ = model_name;
    LOG_INFO("Segmentation model loaded: %s (%s)", model_name.c_str(),
             sam3_is_visual_only(*model_) ? "visual-only" : "full");
}

static void appendPvsResult(const sam3_result& result, SegmentationResult& out) {
    for (const auto& detection : result.detections) {
        const sam3_mask& mask = detection.mask;
        if (mask.data.empty() || mask.width <= 0 || mask.height <= 0) {
            continue;
        }

        sd_image_t image;
        image.width = mask.width;
        image.height = mask.height;
        image.channel = 1;  // grayscale PNG
        image.data = const_cast<unsigned char*>(mask.data.data());

        std::string base64 = imageToBase64(image);
        if (base64.empty()) {
            LOG_WARNING("Failed to encode segmentation mask as PNG, skipping");
            continue;
        }

        out.masks.push_back(std::move(base64));
        out.scores.push_back(detection.score);
    }
}

SegmentationResult ImageSegmenter::segment(const SegmentationParams& params) {
    std::lock_guard<std::mutex> lock(mutex_);

    ensureModelLoaded(params.model_name);

    // Decode the input image as RGB
    sd_image_t input_image = base64ToImage(params.image_base64, 3);
    if (!input_image.data || input_image.width == 0 || input_image.height == 0) {
        throw std::runtime_error("Failed to decode input image");
    }

    struct ImageGuard {
        sd_image_t& img;
        ~ImageGuard() { freeImage(img); }
    } guard{input_image};

    sam3_image image;
    image.width = input_image.width;
    image.height = input_image.height;
    image.channels = 3;
    image.data.assign(input_image.data, input_image.data + (size_t)input_image.width * input_image.height * 3);

    if (!sam3_encode_image(*state_, *model_, image)) {
        throw std::runtime_error("Failed to encode image for segmentation");
    }

    SegmentationResult result;

    bool has_text = !params.text_prompt.empty();
    bool has_points = !params.positive_points.empty() || !params.negative_points.empty();
    bool has_boxes = !params.boxes.empty();

    if (!has_text && !has_points && !has_boxes) {
        throw std::runtime_error(
            "No prompt provided. Specify text_prompt (SAM3 models only), positive/negative_points, or boxes");
    }

    if (has_text) {
        if (sam3_is_visual_only(*model_)) {
            throw std::runtime_error("text_prompt is only supported by full SAM3 models. This model is visual-only "
                                     "(SAM2/SAM2.1/EdgeTAM/SAM3-visual). Use points or boxes instead");
        }

        sam3_pcs_params pcs;
        pcs.text_prompt = params.text_prompt;
        pcs.score_threshold = params.score_threshold;

        sam3_result pcs_result = sam3_segment_pcs(*state_, *model_, pcs);
        appendPvsResult(pcs_result, result);
    }

    if (has_points) {
        sam3_pvs_params pvs;
        for (const auto& [x, y] : params.positive_points) {
            pvs.pos_points.push_back({x, y});
        }
        for (const auto& [x, y] : params.negative_points) {
            pvs.neg_points.push_back({x, y});
        }
        pvs.multimask = false;

        sam3_result pvs_result = sam3_segment_pvs(*state_, *model_, pvs);
        appendPvsResult(pvs_result, result);
    }

    if (has_boxes) {
        for (const auto& box : params.boxes) {
            if (box.size() != 4) {
                throw std::runtime_error("Each box must be [x0, y0, x1, y1]");
            }

            sam3_pvs_params pvs;
            pvs.box = {box[0], box[1], box[2], box[3]};
            pvs.use_box = true;
            pvs.multimask = false;

            sam3_result pvs_result = sam3_segment_pvs(*state_, *model_, pvs);
            appendPvsResult(pvs_result, result);
        }
    }

    LOG_INFO("Segmentation complete: %zu mask(s)", result.masks.size());
    return result;
}
