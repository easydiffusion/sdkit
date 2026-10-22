#ifndef MODEL_DETECTION_H
#define MODEL_DETECTION_H

#include <string>
#include <vector>

// Detect file format by reading magic header
// Returns: "gguf", "safetensors", or "unknown"
std::string detectModelFormat(const std::string& model_path);

// Extract tensor keys from a model file based on its format
// Supports both GGUF and safetensors formats
std::vector<std::string> extractTensorKeys(const std::string& model_path, const std::string& format);

// Determine model type from tensor keys
// Returns: "clip_l", "clip_g", "t5xxl", "llm", "llm_vision", or "vae" (default)
std::string inferModelTypeFromTensorKeys(const std::vector<std::string>& tensor_keys);

// Inspect a model file and determine its type
// Supports both GGUF and safetensors formats
// Returns: "vae", "clip_l", "clip_g", "t5xxl", "llm", "llm_vision", "tokenizer", or "unknown"
std::string inspectModelType(const std::string& model_path);

#endif  // MODEL_DETECTION_H