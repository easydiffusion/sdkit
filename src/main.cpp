#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#else
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "logging.h"
#include "model_manager.h"
#include "server.h"

std::unique_ptr<Server> g_server;
std::atomic<bool> g_should_exit(false);
std::unique_ptr<std::thread> g_watchdog_thread;

void parent_watchdog(int parent_pid) {
    LOG_INFO("Starting parent process watchdog for PID %d", parent_pid);

#ifdef _WIN32
    // Windows: Use timed wait so we can check for exit signal
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, parent_pid);
    if (process == NULL) {
        LOG_ERROR("Failed to open parent process (PID %d). Cannot monitor.", parent_pid);
        return;
    }

    // Wait with timeout so we can check g_should_exit periodically
    while (!g_should_exit.load()) {
        DWORD result = WaitForSingleObject(process, 1000);  // 1 second timeout
        if (result == WAIT_OBJECT_0) {
            // Parent process exited
            LOG_WARNING("Parent process (PID %d) is no longer running. Shutting down...", parent_pid);
            if (g_server) {
                g_server->stop();
            }
            g_should_exit.store(true);
            break;
        }
        // WAIT_TIMEOUT means parent is still alive, continue loop
    }
    CloseHandle(process);
#else
    // Unix/Linux/Mac: Poll with kill(pid, 0) check
    while (!g_should_exit.load()) {
        if (kill(parent_pid, 0) != 0) {
            LOG_WARNING("Parent process (PID %d) is no longer running. Shutting down...", parent_pid);
            if (g_server) {
                g_server->stop();
            }
            g_should_exit.store(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
#endif

    LOG_INFO("Parent process watchdog stopped");
}

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_should_exit.store(true);
    if (g_server) {
        g_server->stop();
    }
}

struct CommandLineArgs {
    int port = 8188;
    std::string log_level = "info";
    int parent_pid = 0;
    std::string ckpt_dir;
    std::string vae_dir;
    std::string hypernetwork_dir;
    std::string gfpgan_models_path;
    std::string realesrgan_models_path;
    std::string lora_dir;
    std::string codeformer_models_path;
    std::string embeddings_dir;
    std::string controlnet_dir;
    std::string text_encoder_dir;
    std::string segmentation_models_path;
    bool vae_on_cpu = false;
    bool vae_tiling = false;
    std::string vae_tile_size;
    bool offload_to_cpu = false;
    bool diffusion_fa = false;
    bool control_net_cpu = false;
    bool clip_on_cpu = false;
    bool chroma_disable_dit_mask = false;
};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --port <port>                      Server port (default: 8188)" << std::endl;
    std::cerr << "  --log-level <level>                Log level: verbose, debug, info, warning, error (default: info)"
              << std::endl;
    std::cerr << "  --parent-pid <pid>                 Parent process PID" << std::endl;
    std::cerr << "  --ckpt-dir <path>                  Checkpoint models directory" << std::endl;
    std::cerr << "  --vae-dir <path>                   VAE models directory" << std::endl;
    std::cerr << "  --hypernetwork-dir <path>          Hypernetwork models directory" << std::endl;
    std::cerr << "  --gfpgan-models-path <path>        GFPGAN models directory" << std::endl;
    std::cerr << "  --realesrgan-models-path <path>    RealESRGAN models directory" << std::endl;
    std::cerr << "  --lora-dir <path>                  LoRA models directory" << std::endl;
    std::cerr << "  --codeformer-models-path <path>    Codeformer models directory" << std::endl;
    std::cerr << "  --embeddings-dir <path>            Embeddings directory" << std::endl;
    std::cerr << "  --controlnet-dir <path>            ControlNet models directory" << std::endl;
    std::cerr << "  --text-encoder-dir <path>          Text encoder models directory" << std::endl;
    std::cerr << "  --segmentation-models-path <path>  Segmentation models directory (SAM 1/2/3 models)" << std::endl;
    std::cerr << "  --vae-on-cpu                       Keep VAE on CPU (default: false)" << std::endl;
    std::cerr << "  --vae-tiling                       Enable VAE tiling (default: false)" << std::endl;
    std::cerr << "  --vae-tile-size <size>             VAE tile size (in pixels), format [X]x[Y] (default: 256x256)"
              << std::endl;
    std::cerr << "  --offload-to-cpu                   Offload parameters to CPU (default: false)" << std::endl;
    std::cerr << "  --diffusion-fa                     Enable diffusion flash attention (default: false)" << std::endl;
    std::cerr << "  --control-net-cpu                  Keep ControlNet on CPU (default: false)" << std::endl;
    std::cerr << "  --clip-on-cpu                      Keep CLIP on CPU (default: false)" << std::endl;
    std::cerr << "  --chroma-disable-dit-mask          Disable DiT mask for Chroma models (default: false)"
              << std::endl;
}

CommandLineArgs parse_args(int argc, char* argv[]) {
    CommandLineArgs args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--port" && i + 1 < argc) {
            try {
                args.port = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid port number: " << argv[i] << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        } else if (arg == "--log-level" && i + 1 < argc) {
            args.log_level = argv[++i];
        } else if (arg == "--parent-pid" && i + 1 < argc) {
            try {
                args.parent_pid = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid parent PID: " << argv[i] << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        } else if (arg == "--ckpt-dir" && i + 1 < argc) {
            args.ckpt_dir = argv[++i];
        } else if (arg == "--vae-dir" && i + 1 < argc) {
            args.vae_dir = argv[++i];
        } else if (arg == "--hypernetwork-dir" && i + 1 < argc) {
            args.hypernetwork_dir = argv[++i];
        } else if (arg == "--gfpgan-models-path" && i + 1 < argc) {
            args.gfpgan_models_path = argv[++i];
        } else if (arg == "--realesrgan-models-path" && i + 1 < argc) {
            args.realesrgan_models_path = argv[++i];
        } else if (arg == "--lora-dir" && i + 1 < argc) {
            args.lora_dir = argv[++i];
        } else if (arg == "--codeformer-models-path" && i + 1 < argc) {
            args.codeformer_models_path = argv[++i];
        } else if (arg == "--embeddings-dir" && i + 1 < argc) {
            args.embeddings_dir = argv[++i];
        } else if (arg == "--controlnet-dir" && i + 1 < argc) {
            args.controlnet_dir = argv[++i];
        } else if (arg == "--text-encoder-dir" && i + 1 < argc) {
            args.text_encoder_dir = argv[++i];
        } else if (arg == "--segmentation-models-path" && i + 1 < argc) {
            args.segmentation_models_path = argv[++i];
        } else if (arg == "--vae-on-cpu") {
            args.vae_on_cpu = true;
        } else if (arg == "--vae-tiling") {
            args.vae_tiling = true;
        } else if (arg == "--vae-tile-size" && i + 1 < argc) {
            args.vae_tile_size = argv[++i];
        } else if (arg == "--offload-to-cpu") {
            args.offload_to_cpu = true;
        } else if (arg == "--diffusion-fa") {
            args.diffusion_fa = true;
        } else if (arg == "--control-net-cpu") {
            args.control_net_cpu = true;
        } else if (arg == "--clip-on-cpu") {
            args.clip_on_cpu = true;
        } else if (arg == "--chroma-disable-dit-mask") {
            args.chroma_disable_dit_mask = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    return args;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Parse command line arguments
    CommandLineArgs args = parse_args(argc, argv);

    // Set log level from command line argument
    set_log_level(args.log_level);

    // Create and configure model manager
    auto model_manager = std::make_shared<ModelManager>();

    if (!args.ckpt_dir.empty()) {
        model_manager->setCheckpointDir(args.ckpt_dir);
    }
    if (!args.vae_dir.empty()) {
        model_manager->setVaeDir(args.vae_dir);
    }
    if (!args.hypernetwork_dir.empty()) {
        model_manager->setHypernetworkDir(args.hypernetwork_dir);
    }
    if (!args.gfpgan_models_path.empty()) {
        model_manager->setGfpganModelsPath(args.gfpgan_models_path);
    }
    if (!args.realesrgan_models_path.empty()) {
        model_manager->setRealesrganModelsPath(args.realesrgan_models_path);
    }
    if (!args.lora_dir.empty()) {
        model_manager->setLoraDir(args.lora_dir);
    }
    if (!args.codeformer_models_path.empty()) {
        model_manager->setCodeformerModelsPath(args.codeformer_models_path);
    }
    if (!args.embeddings_dir.empty()) {
        model_manager->setEmbeddingsDir(args.embeddings_dir);
    }
    if (!args.controlnet_dir.empty()) {
        model_manager->setControlnetDir(args.controlnet_dir);
    }
    if (!args.text_encoder_dir.empty()) {
        model_manager->setTextEncoderDir(args.text_encoder_dir);
    }
    if (!args.segmentation_models_path.empty()) {
        model_manager->setSegmentationDir(args.segmentation_models_path);
    }

    // Scan all model directories
    std::cout << "Scanning model directories..." << std::endl;
    model_manager->scanAllDirectories();
    std::cout << "Model scanning complete." << std::endl;
    std::cout << std::endl;

    // Start parent process watchdog if parent PID is specified
    if (args.parent_pid > 0) {
        g_watchdog_thread = std::make_unique<std::thread>(parent_watchdog, args.parent_pid);
    }

    try {
        // Create server parameters
        ServerParams server_params;
        server_params.port = args.port;
        server_params.model_manager = model_manager;
        server_params.vae_on_cpu = args.vae_on_cpu;
        server_params.vae_tiling = args.vae_tiling;
        server_params.vae_tile_size = args.vae_tile_size;
        server_params.offload_to_cpu = args.offload_to_cpu;
        server_params.diffusion_fa = args.diffusion_fa;
        server_params.control_net_cpu = args.control_net_cpu;
        server_params.clip_on_cpu = args.clip_on_cpu;
        server_params.chroma_disable_dit_mask = args.chroma_disable_dit_mask;

        // Create and start the server
        g_server = std::make_unique<Server>(server_params);
        g_server->run();
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        g_should_exit.store(true);
        if (g_watchdog_thread && g_watchdog_thread->joinable()) {
            g_watchdog_thread->join();
        }
        return 1;
    }

    // Clean up watchdog thread
    g_should_exit.store(true);
    if (g_watchdog_thread && g_watchdog_thread->joinable()) {
        g_watchdog_thread->join();
    }

    std::cout << "Server stopped." << std::endl;
    return 0;
}
