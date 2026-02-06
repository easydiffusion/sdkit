#include "string_utils.h"

#include <codecvt>
#include <locale>

#ifdef _WIN32
#include <windows.h>
#endif

// UTF-8 conversion utilities
#ifdef _WIN32
// Windows-specific UTF-8 to UTF-16 conversion
std::wstring utf8_to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    if (size_needed <= 0) return std::wstring();
    std::wstring result(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &result[0], size_needed);
    return result;
}

// Windows-specific UTF-16 to UTF-8 conversion
std::string wstring_to_utf8(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
    if (size_needed <= 0) return std::string();
    std::string result(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &result[0], size_needed, NULL, NULL);
    return result;
}
#else
// POSIX systems (Linux, macOS) - use codecvt (UTF-8 is typically native)
std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t> > converter;
    return converter.from_bytes(str);
}

std::string wstring_to_utf8(const std::wstring& wstr) {
    std::wstring_convert<std::codecvt_utf8<wchar_t> > converter;
    return converter.to_bytes(wstr);
}
#endif

// Cross-platform path to UTF-8 string conversion
std::string path_to_utf8(const fs::path& path) {
#ifdef _WIN32
    // On Windows, use native() to get wstring, then convert to UTF-8
    return wstring_to_utf8(path.wstring());
#else
    // On POSIX, string() already returns UTF-8
    return path.string();
#endif
}
