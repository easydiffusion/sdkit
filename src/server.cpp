#include "server.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "image_utils.h"
#include "logging.h"

// Custom log handler for Crow that filters out /ping requests
class FilteredLogHandler : public crow::ILogHandler {
public:
  void log(std::string message, crow::LogLevel level) override {
    // Skip logging /ping requests to reduce noise
    if (message.find("/ping") != std::string::npos ||
        message.find("/internal/progress") != std::string::npos) {
      return;
    }

    // Use our existing logging system for consistency
    LogLevel our_level;
    switch (level) {
    case crow::LogLevel::Debug:
      our_level = LogLevel::Debug;
      break;
    case crow::LogLevel::Info:
      our_level = LogLevel::Info;
      break;
    case crow::LogLevel::Warning:
      our_level = LogLevel::Warning;
      break;
    case crow::LogLevel::Error:
    case crow::LogLevel::Critical:
      our_level = LogLevel::Error;
      break;
    default:
      our_level = LogLevel::Info;
      break;
    }

    log_message(our_level, "[CROW] %s", message.c_str());
  }
};

// Convert webui (Forge/Automatic1111 style) sampler/scheduler names
// into stable-diffusion.cpp compatible names.
static std::string convert_webui_sampler_name(const std::string &name) {
  static const std::unordered_map<std::string, std::string> mapping = {
      {"Euler", "euler"},
      {"Euler a", "euler_a"},
      {"Heun", "heun"},
      {"DPM2", "dpm2"},
      {"DPM++ 2S a", "dpm++2s_a"},
      {"DPM++ 2M", "dpm++2m"},
      {"DPM++ 2M v2", "dpm++2mv2"},
      {"IPNDM", "ipndm"},
      {"IPNDM_V", "ipndm_v"},
      {"LCM", "lcm"},
      {"DDIM", "ddim_trailing"},
      {"TCD", "tcd"},
  };

  auto it = mapping.find(name);
  if (it != mapping.end())
    return it->second;
  return name;
}

static std::string convert_webui_scheduler_name(const std::string &name) {
  static const std::unordered_map<std::string, std::string> mapping = {
      {"automatic", "discrete"},      {"uniform", "discrete"},
      {"karras", "karras"},           {"exponential", "exponential"},
      {"sgm_uniform", "sgm_uniform"}, {"simple", "simple"},
      {"align_your_steps", "ays"},    {"align_your_steps_GITS", "gits"},
  };

  auto it = mapping.find(name);
  if (it != mapping.end())
    return it->second;
  return name;
}

// Convert webui upscaler names to stable-diffusion.cpp compatible names
static std::string convert_webui_upscaler_name(const std::string &name) {
  static const std::unordered_map<std::string, std::string> mapping = {
      {"R-ESRGAN 4x+", "RealESRGAN_x4plus"},
      {"R-ESRGAN 4x+ Anime6B", "RealESRGAN_x4plus_anime_6B"},
      // Add more mappings as needed
  };

  auto it = mapping.find(name);
  if (it != mapping.end())
    return it->second;
  return name;
}

// Convert webui ControlNet model names to sdkit compatible names (remove hash
// suffix)
static std::string
convert_webui_controlnet_model_name(const std::string &name) {
  // Remove the hash suffix like " [a3cd7cd6]" from the end
  size_t bracket_pos = name.find_last_of('[');
  if (bracket_pos != std::string::npos) {
    return name.substr(0, bracket_pos - 1); // -1 to remove the space before [
  }
  return name;
}

// Round dimension to nearest multiple of 64
static int round_to_nearest_multiple_of_64(int dimension) {
  return std::round(static_cast<double>(dimension) / 64.0) * 64;
}

Server::Server(const ServerParams &params)
    : params_(params), port_(params.port), model_manager_(params.model_manager),
      should_stop_(false) {
  // Set up custom logger to filter out unnecessary requests
  static FilteredLogHandler filtered_handler;
  crow::logger::setHandler(&filtered_handler);

  options_manager_ = std::make_shared<OptionsManager>();
  task_state_manager_ = std::make_shared<TaskStateManager>();

  // Create ImageFilters for upscaling and other image processing
  image_filters_ = std::make_shared<ImageFilters>(model_manager_);

  // Create ImageSegmenter for SAM-based segmentation
  image_segmenter_ = std::make_unique<ImageSegmenter>(model_manager_);

  // Create ImageGenerator with shared task state manager and model manager
  image_generator_ =
      std::make_unique<ImageGenerator>(task_state_manager_, options_manager_,
                                       model_manager_, image_filters_, params);

  options_manager_->load();

  setupRoutes();
}

Server::~Server() { stop(); }

void Server::setupRoutes() {
  // Ping endpoint
  CROW_ROUTE(app_, "/v1/internal/ping").methods("GET"_method)([this]() {
    return handlePing();
  });

  // Demo pages (served same-origin so the browser allows API calls)
  CROW_ROUTE(app_, "/").methods("GET"_method)(
      [this]() { return handleDemoPage("index.html"); });

  CROW_ROUTE(app_, "/demo/<string>")
      .methods("GET"_method)(
          [this](const std::string &name) { return handleDemoPage(name); });

  // Models endpoint
  CROW_ROUTE(app_, "/v1/sdapi/v1/models").methods("GET"_method)([this]() {
    return handleGetModels();
  });

  // Options endpoints
  CROW_ROUTE(app_, "/v1/sdapi/v1/options").methods("GET"_method)([this]() {
    return handleGetOptions();
  });

  CROW_ROUTE(app_, "/v1/sdapi/v1/options")
      .methods("POST"_method)(
          [this](const crow::request &req) { return handlePostOptions(req); });

  // Image generation endpoints
  CROW_ROUTE(app_, "/v1/sdapi/v1/txt2img")
      .methods("POST"_method)(
          [this](const crow::request &req) { return handleTxt2Img(req); });

  CROW_ROUTE(app_, "/v1/sdapi/v1/img2img")
      .methods("POST"_method)(
          [this](const crow::request &req) { return handleImg2Img(req); });

  // Progress endpoint
  CROW_ROUTE(app_, "/v1/internal/progress")
      .methods("POST"_method)(
          [this](const crow::request &req) { return handleProgress(req); });

  // Interrupt endpoint
  CROW_ROUTE(app_, "/v1/sdapi/v1/interrupt")
      .methods("POST"_method)(
          [this](const crow::request &req) { return handleInterrupt(req); });

  // Extra batch images endpoint
  CROW_ROUTE(app_, "/v1/sdapi/v1/extra-batch-images")
      .methods("POST"_method)([this](const crow::request &req) {
        return handleExtraBatchImages(req);
      });

  // ControlNet detect endpoint
  CROW_ROUTE(app_, "/v1/controlnet/detect")
      .methods("POST"_method)([this](const crow::request &req) {
        return handleControlNetDetect(req);
      });

  // Segmentation endpoint
  CROW_ROUTE(app_, "/v1/sdapi/v1/segment")
      .methods("POST"_method)(
          [this](const crow::request &req) { return handleSegment(req); });

  // Refresh endpoints
  CROW_ROUTE(app_, "/v1/sdapi/v1/refresh-checkpoints")
      .methods("POST"_method)([this]() { return handleRefreshCheckpoints(); });

  CROW_ROUTE(app_, "/v1/sdapi/v1/refresh-vae-and-text-encoders")
      .methods("POST"_method)(
          [this]() { return handleRefreshVaeAndTextEncoders(); });
}

void Server::run() {
  std::cout << "Starting server on port " << port_ << std::endl;
  app_.bindaddr("127.0.0.1").port(port_).multithreaded().run();
}

void Server::stop() {
  should_stop_ = true;
  app_.stop();
}

crow::response Server::handlePing() { return crow::response(200, "OK"); }

crow::response Server::handleDemoPage(const std::string &filename) {
  // Only serve plain .html files from within the demo directory.
  if (filename.empty() || filename.find("/") != std::string::npos ||
      filename.find("\\") != std::string::npos ||
      filename.find("..") != std::string::npos || filename.length() < 5 ||
      filename.substr(filename.length() - 5) != ".html") {
    return crow::response(404, "Demo page not found");
  }

  // Look for the demo page relative to the working directory, then relative to
  // the executable's directory (so it works regardless of where the server is
  // started from).
  std::vector<std::string> candidates;
  candidates.push_back("demo/" + filename);

  char exe_path_buf[MAX_PATH];
  if (GetModuleFileNameA(nullptr, exe_path_buf, MAX_PATH) > 0) {
    std::string exe_dir(exe_path_buf);
    size_t slash = exe_dir.find_last_of("/\\");
    if (slash != std::string::npos) {
      candidates.push_back(exe_dir.substr(0, slash + 1) + "demo/" + filename);
    }
  }

  for (const auto &path : candidates) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
      continue;
    std::ostringstream ss;
    ss << file.rdbuf();
    crow::response res(200, ss.str());
    res.set_header("Content-Type", "text/html; charset=utf-8");
    return res;
  }

  LOG_ERROR("Demo page not found (looked in: %s)",
            ("demo/" + filename).c_str());
  return crow::response(
      404,
      "Demo page not found. Start sdkit.exe from the repository root, or place"
      " the demo/ directory next to the executable.");
}

crow::response Server::handleGetModels() {
  try {
    auto grouped_models = model_manager_->getAllModelsGrouped();

    crow::json::wvalue response;
    for (const auto &[type, names] : grouped_models) {
      response[type] = names;
    }

    return crow::response(200, response);
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] = std::string("Failed to get models: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleGetOptions() {
  try {
    auto options = options_manager_->getOptions();
    return crow::response(200, options);
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] = std::string("Failed to get options: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handlePostOptions(const crow::request &req) {
  try {
    auto json_body = crow::json::load(req.body);
    if (!json_body) {
      crow::json::wvalue error;
      error["message"] = "Invalid JSON";
      return crow::response(400, error);
    }

    if (options_manager_->setOptions(json_body)) {
      return crow::response(200, "OK");
    } else {
      crow::json::wvalue error;
      error["message"] = "Failed to save options";
      return crow::response(500, error);
    }
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] = std::string("Failed to set options: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleTxt2Img(const crow::request &req) {
  try {
    auto json_body = crow::json::load(req.body);
    if (!json_body) {
      crow::json::wvalue error;
      error["message"] = "Invalid JSON";
      return crow::response(400, error);
    }

    return generateImage(json_body, false);
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] = std::string("Failed to generate image: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleImg2Img(const crow::request &req) {
  try {
    auto json_body = crow::json::load(req.body);
    if (!json_body) {
      crow::json::wvalue error;
      error["message"] = "Invalid JSON";
      return crow::response(400, error);
    }

    return generateImage(json_body, true);
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] = std::string("Failed to generate image: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::generateImage(const crow::json::rvalue &json_body,
                                     bool is_img2img) {
  // Extract task_id
  std::string task_id = "default_task";
  if (json_body.has("force_task_id")) {
    task_id = json_body["force_task_id"].s();
  }

  // Create task
  task_state_manager_->createTask(task_id);

  try {
    // Parse generation parameters
    ImageGenerationParams params;
    params.prompt =
        json_body.has("prompt") ? std::string(json_body["prompt"].s()) : "";
    params.negative_prompt = json_body.has("negative_prompt")
                                 ? std::string(json_body["negative_prompt"].s())
                                 : "";
    if (json_body.has("lora_paths") &&
        json_body["lora_paths"].t() == crow::json::type::List) {
      for (size_t i = 0; i < json_body["lora_paths"].size(); i++) {
        params.lora_paths.push_back(
            std::string(json_body["lora_paths"][i].s()));
      }
    }
    if (json_body.has("lora_alphas") &&
        json_body["lora_alphas"].t() == crow::json::type::List) {
      for (size_t i = 0; i < json_body["lora_alphas"].size(); i++) {
        params.lora_alphas.push_back(
            static_cast<float>(json_body["lora_alphas"][i].d()));
      }
    }
    params.width = json_body.has("width") ? json_body["width"].i() : 512;
    params.height = json_body.has("height") ? json_body["height"].i() : 512;
    params.steps = json_body.has("steps") ? json_body["steps"].i() : 20;
    params.cfg_scale =
        json_body.has("cfg_scale") ? json_body["cfg_scale"].d() : 7.0f;
    params.seed = json_body.has("seed") ? json_body["seed"].i() : -1;
    params.batch_count =
        json_body.has("batch_size") ? json_body["batch_size"].i() : 1;

    // Sampler and scheduler parameters
    if (json_body.has("sampler_name") &&
        json_body["sampler_name"].t() == crow::json::type::String) {
      std::string sampler_str = json_body["sampler_name"].s();
      // Convert from webui-style sampler name to sd.cpp name
      sampler_str = convert_webui_sampler_name(sampler_str);
      params.sampler = str_to_sample_method(sampler_str.c_str());
    }
    if (json_body.has("scheduler") &&
        json_body["scheduler"].t() == crow::json::type::String) {
      std::string scheduler_str = json_body["scheduler"].s();
      // Convert from webui-style scheduler name to sd.cpp name
      scheduler_str = convert_webui_scheduler_name(scheduler_str);
      params.scheduler = str_to_scheduler(scheduler_str.c_str());
    }

    // img2img specific parameters
    if (is_img2img) {
      if (json_body.has("init_images") && json_body["init_images"].size() > 0) {
        params.init_image_base64 = std::string(json_body["init_images"][0].s());
      }
      if (json_body.has("mask")) {
        params.mask_base64 = std::string(json_body["mask"].s());
      }
      params.strength = json_body.has("denoising_strength")
                            ? json_body["denoising_strength"].d()
                            : 0.75f;
    }

    // reference images
    if (json_body.has("ref_images") && json_body["ref_images"].size() > 0) {
      for (size_t i = 0; i < json_body["ref_images"].size(); i++) {
        params.ref_images_base64.push_back(
            std::string(json_body["ref_images"][i].s()));
      }
    }

    // ControlNet parameters from alwayson_scripts
    if (json_body.has("alwayson_scripts") &&
        json_body["alwayson_scripts"].has("controlnet")) {
      auto controlnet_obj = json_body["alwayson_scripts"]["controlnet"];
      if (controlnet_obj.has("args") && controlnet_obj["args"].size() > 0) {
        auto controlnet_args = controlnet_obj["args"][0];

        // Extract control image
        if (controlnet_args.has("image")) {
          params.control_image_base64 =
              std::string(controlnet_args["image"].s());
        }

        // Extract control strength (weight)
        if (controlnet_args.has("weight")) {
          params.control_strength = controlnet_args["weight"].d();
        }

        // Extract controlnet model name
        if (controlnet_args.has("model")) {
          std::string webui_model_name =
              std::string(controlnet_args["model"].s());
          params.controlnet_model =
              convert_webui_controlnet_model_name(webui_model_name);
        }

        LOG_INFO("ControlNet params: model='%s', strength=%.2f, has_image=%s",
                 params.controlnet_model.c_str(), params.control_strength,
                 params.control_image_base64.empty() ? "no" : "yes");
      }
    }

    // Round dimensions to nearest multiple of 64 when ControlNet is used
    if (!params.controlnet_model.empty()) {
      int original_width = params.width;
      int original_height = params.height;
      params.width = round_to_nearest_multiple_of_64(params.width);
      params.height = round_to_nearest_multiple_of_64(params.height);
      LOG_INFO("ControlNet detected, rounded dimensions from %dx%d to %dx%d",
               original_width, original_height, params.width, params.height);
    }

    // Generate images (runs in same thread, blocks until complete)
    std::vector<std::string> images;
    if (is_img2img) {
      images = image_generator_->generateImg2Img(params, task_id);
    } else {
      images = image_generator_->generateTxt2Img(params, task_id);
    }

    // Create info JSON string
    crow::json::wvalue info_json;
    info_json["prompt"] = params.prompt;
    info_json["negative_prompt"] = params.negative_prompt;
    info_json["steps"] = params.steps;
    info_json["cfg_scale"] = params.cfg_scale;
    info_json["seed"] = params.seed;
    info_json["width"] = params.width;
    info_json["height"] = params.height;

    crow::json::wvalue infotexts_json;
    infotexts_json["infotexts"] = info_json.dump();
    std::string info = infotexts_json.dump();

    // Complete task
    task_state_manager_->completeTask(task_id, images, info);

    // Return response
    crow::json::wvalue response;
    response["images"] = images;
    response["info"] = info;

    return crow::response(200, response);

  } catch (const std::exception &e) {
    LOG_ERROR("Image generation error: %s", e.what());
    crow::json::wvalue error;
    error["message"] = std::string("Generation failed: ") + e.what();
    task_state_manager_->completeTask(task_id, {}, error.dump());
    return crow::response(500, error);
  }
}

crow::response Server::handleProgress(const crow::request &req) {
  try {
    auto json_body = crow::json::load(req.body);
    if (!json_body) {
      crow::json::wvalue error;
      error["message"] = "Invalid JSON";
      return crow::response(400, error);
    }

    if (!json_body.has("id_task")) {
      crow::json::wvalue error;
      error["message"] = "Missing id_task parameter";
      return crow::response(400, error);
    }

    std::string task_id = json_body["id_task"].s();

    if (!task_state_manager_->taskExists(task_id)) {
      crow::json::wvalue error;
      error["message"] = "Task not found";
      return crow::response(404, error);
    }

    TaskState state = task_state_manager_->getTaskState(task_id);

    crow::json::wvalue response;
    response["completed"] = state.completed;
    response["progress"] = state.progress;
    response["live_preview"] = state.live_preview;
    response["id_live_preview"] = state.id_live_preview;

    return crow::response(200, response);
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] = std::string("Failed to get progress: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleInterrupt(const crow::request &req) {
  try {
    // Interrupt the image generator
    if (image_generator_) {
      image_generator_->interrupt();
      LOG_INFO("Image generation interrupted");
    }

    return crow::response(200, "OK");
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] = std::string("Failed to interrupt: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleExtraBatchImages(const crow::request &req) {
  try {
    auto json_body = crow::json::load(req.body);
    if (!json_body) {
      crow::json::wvalue error;
      error["message"] = "Invalid JSON";
      return crow::response(400, error);
    }

    if (!json_body.has("imageList")) {
      crow::json::wvalue error;
      error["message"] = "Missing imageList parameter";
      return crow::response(400, error);
    }

    // Check if upscaling is requested
    int upscaling_resize = json_body.has("upscaling_resize")
                               ? json_body["upscaling_resize"].i()
                               : 0;

    // Get upscaler name from request (defaults to empty string)
    std::string upscaler_name;
    if (json_body.has("upscaler_1")) {
      std::string webui_upscaler_name = json_body["upscaler_1"].s();
      upscaler_name = convert_webui_upscaler_name(webui_upscaler_name);
    }

    auto image_list = json_body["imageList"];
    std::vector<std::string> result_images;

    if (upscaling_resize > 0) {
      // Collect images into vector
      std::vector<std::string> input_images;
      for (size_t i = 0; i < image_list.size(); i++) {
        input_images.push_back(image_list[i]["data"].s());
      }

      LOG_INFO(
          "Upscaling %zu images with upscaling factor %d using upscaler: %s",
          input_images.size(), upscaling_resize, upscaler_name.c_str());

      // Use ImageFilters to upscale with specified upscaler
      result_images = image_filters_->upscaleBatch(input_images, upscaler_name,
                                                   upscaling_resize);
      if (result_images.empty()) {
        crow::json::wvalue error;
        error["message"] = "Upscaler not available. Please configure an "
                           "upscaler model in options.";
        return crow::response(500, error);
      }
    } else {
      // No upscaling requested, just return the images as-is
      for (size_t i = 0; i < image_list.size(); i++) {
        result_images.push_back(image_list[i]["data"].s());
      }
    }

    crow::json::wvalue response;
    response["images"] = result_images;

    return crow::response(200, response);
  } catch (const std::exception &e) {
    LOG_ERROR("Extra batch images error: %s", e.what());

    crow::json::wvalue error;
    error["message"] = std::string("Failed to process images: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleControlNetDetect(const crow::request &req) {
  try {
    auto json_body = crow::json::load(req.body);
    if (!json_body) {
      crow::json::wvalue error;
      error["message"] = "Invalid JSON";
      return crow::response(400, error);
    }

    std::vector<std::string> result_images;

    if (json_body.has("controlnet_input_images")) {
      auto input_images = json_body["controlnet_input_images"];
      std::vector<std::string> base64_images;
      for (size_t i = 0; i < input_images.size(); i++) {
        base64_images.push_back(std::string(input_images[i].s()));
      }

      std::string module = "canny";
      if (json_body.has("controlnet_module")) {
        module = json_body["controlnet_module"].s();
      }

      // Use ImageFilters to apply ControlNet preprocessing
      result_images =
          image_filters_->applyControlNetFilterBatch(base64_images, module);
    }

    crow::json::wvalue response;
    response["images"] = result_images;

    return crow::response(200, response);
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] = std::string("Failed to detect: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleSegment(const crow::request &req) {
  try {
    auto json_body = crow::json::load(req.body);
    if (!json_body) {
      crow::json::wvalue error;
      error["message"] = "Invalid JSON";
      return crow::response(400, error);
    }

    SegmentationParams params;

    if (json_body.has("image") &&
        json_body["image"].t() == crow::json::type::String) {
      params.image_base64 = json_body["image"].s();
    }
    if (params.image_base64.empty()) {
      crow::json::wvalue error;
      error["message"] = "Missing image parameter";
      return crow::response(400, error);
    }

    if (json_body.has("model") &&
        json_body["model"].t() == crow::json::type::String) {
      params.model_name = json_body["model"].s();
    }
    if (params.model_name.empty()) {
      crow::json::wvalue error;
      error["message"] = "Missing model parameter";
      return crow::response(400, error);
    }

    if (json_body.has("text_prompt") &&
        json_body["text_prompt"].t() == crow::json::type::String) {
      params.text_prompt = json_body["text_prompt"].s();
    }
    if (json_body.has("score_threshold") &&
        json_body["score_threshold"].t() == crow::json::type::Number) {
      params.score_threshold =
          static_cast<float>(json_body["score_threshold"].d());
    }

    auto read_points =
        [&json_body](const char *key,
                     std::vector<std::pair<float, float>> &out) -> bool {
      if (!json_body.has(key))
        return true;
      if (json_body[key].t() != crow::json::type::List)
        return false;
      for (size_t i = 0; i < json_body[key].size(); i++) {
        auto &point = json_body[key][i];
        if (point.t() != crow::json::type::List || point.size() != 2)
          return false;
        out.emplace_back(static_cast<float>(point[0].d()),
                         static_cast<float>(point[1].d()));
      }
      return true;
    };

    if (!read_points("positive_points", params.positive_points) ||
        !read_points("negative_points", params.negative_points)) {
      crow::json::wvalue error;
      error["message"] = "Points must be lists of [x, y] pairs";
      return crow::response(400, error);
    }

    if (json_body.has("boxes")) {
      if (json_body["boxes"].t() != crow::json::type::List) {
        crow::json::wvalue error;
        error["message"] = "boxes must be a list of [x0, y0, x1, y1] arrays";
        return crow::response(400, error);
      }
      for (size_t i = 0; i < json_body["boxes"].size(); i++) {
        auto &box = json_body["boxes"][i];
        if (box.t() != crow::json::type::List || box.size() != 4) {
          crow::json::wvalue error;
          error["message"] = "Each box must be [x0, y0, x1, y1]";
          return crow::response(400, error);
        }
        std::vector<float> coords;
        for (size_t j = 0; j < 4; j++) {
          coords.push_back(static_cast<float>(box[j].d()));
        }
        params.boxes.push_back(std::move(coords));
      }
    }

    LOG_INFO("Segmenting image with model '%s' (prompt: %s, %zu positive "
             "point(s), %zu negative point(s), %zu "
             "box(es))",
             params.model_name.c_str(),
             params.text_prompt.empty() ? "none" : params.text_prompt.c_str(),
             params.positive_points.size(), params.negative_points.size(),
             params.boxes.size());

    SegmentationResult result = image_segmenter_->segment(params);

    crow::json::wvalue response;
    response["masks"] = result.masks;
    response["scores"] = result.scores;

    return crow::response(200, response);
  } catch (const std::exception &e) {
    LOG_ERROR("Segmentation error: %s", e.what());
    crow::json::wvalue error;
    error["message"] = std::string("Segmentation failed: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleRefreshCheckpoints() {
  try {
    LOG_INFO("Refreshing checkpoints...");
    if (model_manager_) {
      model_manager_->refreshCheckpoints();
    }
    return crow::response(200, "OK");
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] =
        std::string("Failed to refresh checkpoints: ") + e.what();
    return crow::response(500, error);
  }
}

crow::response Server::handleRefreshVaeAndTextEncoders() {
  try {
    LOG_INFO("Refreshing VAE and text encoders...");
    if (model_manager_) {
      model_manager_->refreshVaeAndTextEncoders();
    }
    return crow::response(200, "OK");
  } catch (const std::exception &e) {
    crow::json::wvalue error;
    error["message"] =
        std::string("Failed to refresh VAE and text encoders: ") + e.what();
    return crow::response(500, error);
  }
}
