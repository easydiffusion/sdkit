#ifndef __MODEL_MANAGER_H__
#define __MODEL_MANAGER_H__

#include <map>
#include <mutex>
#include <string>
#include <vector>

enum class ModelType {
    CHECKPOINT,
    VAE,
    HYPERNETWORK,
    GFPGAN,
    REALESRGAN,
    LORA,
    CODEFORMER,
    EMBEDDINGS,
    CONTROLNET,
    TEXT_ENCODER
};

struct ModelInfo {
    std::string filename;   // Lookup name: relative path with extension (checkpoint), filename without extension
                            // (controlnet/embedding/lora), or filename (others)
    std::string full_path;  // Absolute path to the model file
    ModelType type;
    size_t file_size;  // File size in bytes

    ModelInfo() : type(ModelType::CHECKPOINT), file_size(0) {}
    ModelInfo(const std::string& fname, const std::string& path, ModelType t, size_t size)
        : filename(fname), full_path(path), type(t), file_size(size) {}
};

class ModelManager {
   public:
    ModelManager();
    ~ModelManager();

    // Set model directories
    void setCheckpointDir(const std::string& dir);
    void setVaeDir(const std::string& dir);
    void setHypernetworkDir(const std::string& dir);
    void setGfpganModelsPath(const std::string& dir);
    void setRealesrganModelsPath(const std::string& dir);
    void setLoraDir(const std::string& dir);
    void setCodeformerModelsPath(const std::string& dir);
    void setEmbeddingsDir(const std::string& dir);
    void setControlnetDir(const std::string& dir);
    void setTextEncoderDir(const std::string& dir);

    // Get model directories
    std::string getRealesrganModelsPath() const;
    std::string getLoraDir() const;
    std::string getEmbeddingsDir() const;

    // Scan directories and build the model index
    void scanAllDirectories();
    void scanDirectory(ModelType type);

    // Refresh the model index
    void refresh();
    void refreshCheckpoints();
    void refreshVaeAndTextEncoders();

    // Query models
    std::vector<ModelInfo> getModelsByType(ModelType type) const;
    std::vector<std::string> getModelNamesByType(ModelType type) const;
    ModelInfo getModelByName(const std::string& name, ModelType type) const;
    bool hasModel(const std::string& name, ModelType type) const;

    // Get all models as JSON-friendly structure
    std::map<std::string, std::vector<std::string>> getAllModelsGrouped() const;

   private:
    // Directory paths for each model type
    std::map<ModelType, std::string> model_directories_;

    // Indexed models: ModelType -> vector of ModelInfo
    std::map<ModelType, std::vector<ModelInfo>> models_;

    // Mutex for thread-safe access
    mutable std::mutex mutex_;

    // Valid model file extensions
    static const std::vector<std::string> valid_extensions_;

    // Helper methods
    bool isValidModelFile(const std::string& filename) const;
    void scanDirectoryInternal(const std::string& directory, ModelType type);
    std::string getModelTypeString(ModelType type) const;
};

#endif  // __MODEL_MANAGER_H__
