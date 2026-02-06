#pragma once

#include <filesystem>
#include <string>

namespace fs = std::filesystem;

// UTF-8 conversion utilities for cross-platform support
#ifdef _WIN32
// Windows-specific UTF-8 to UTF-16 conversion
std::wstring utf8_to_wstring(const std::string& str);

// Windows-specific UTF-16 to UTF-8 conversion
std::string wstring_to_utf8(const std::wstring& wstr);
#else
// POSIX systems (Linux, macOS) - use codecvt (UTF-8 is typically native)
std::wstring utf8_to_wstring(const std::string& str);

std::string wstring_to_utf8(const std::wstring& wstr);
#endif

// Cross-platform path to UTF-8 string conversion
std::string path_to_utf8(const fs::path& path);
