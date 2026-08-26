#ifndef __SERVER_H__
#define __SERVER_H__

#include <memory>
#include <string>

#include "crow.h"
#include "image_filters.h"
#include "image_generator.h"
#include "image_segmenter.h"
#include "model_manager.h"
#include "options_manager.h"
#include "task_state.h"

struct ServerParams {
    int port = 8188;
    std::shared_ptr<ModelManager> model_manager;
    bool vae_on_cpu = false;
    bool vae_tiling = false;
    std::string vae_tile_size;
    bool offload_to_cpu = false;
    bool diffusion_fa = false;
    bool control_net_cpu = false;
    bool clip_on_cpu = false;
    bool chroma_disable_dit_mask = false;
};

class Server {
   public:
    Server(const ServerParams& params);
    ~Server();

    void run();
    void stop();

   private:
    void setupRoutes();

    // Route handlers
    crow::response handlePing();
    crow::response handleDemoPage(const std::string& filename);
    crow::response handleGetModels();
    crow::response handleGetOptions();
    crow::response handlePostOptions(const crow::request& req);
    crow::response handleTxt2Img(const crow::request& req);
    crow::response handleImg2Img(const crow::request& req);
    crow::response handleProgress(const crow::request& req);
    crow::response handleInterrupt(const crow::request& req);
    crow::response handleExtraBatchImages(const crow::request& req);
    crow::response handleControlNetDetect(const crow::request& req);
    crow::response handleSegment(const crow::request& req);
    crow::response handleRefreshCheckpoints();
    crow::response handleRefreshVaeAndTextEncoders();

    // Helper methods
    crow::response generateImage(const crow::json::rvalue& json_body, bool is_img2img);

    ServerParams params_;
    int port_;
    crow::SimpleApp app_;
    std::shared_ptr<OptionsManager> options_manager_;
    std::shared_ptr<TaskStateManager> task_state_manager_;
    std::shared_ptr<ModelManager> model_manager_;
    std::unique_ptr<ImageGenerator> image_generator_;
    std::shared_ptr<ImageFilters> image_filters_;
    std::unique_ptr<ImageSegmenter> image_segmenter_;
    bool should_stop_;
};

#endif  // __SERVER_H__
