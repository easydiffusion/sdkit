cmake_minimum_required(VERSION 3.16)
project(sdkit)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable versioned sonames on platforms that support it
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# Option to build tools
option(SDKIT_BUILD_TOOLS "Build sdkit tools" OFF)

# Define output directories for all build artifacts
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set flags for shared library builds
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build ggml as a shared library")
set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "Build stable-diffusion as a shared library")
set(SD_BUILD_SHARED_GGML_LIB ON CACHE BOOL "Build ggml as a separate shared library")

# Enable /bigobj for MSVC to handle large object files (needed for debug builds)
if(MSVC)
    add_compile_options($<$<CONFIG:Debug>:/bigobj>)
endif()

# Include stable-diffusion.cpp
add_subdirectory(stable-diffusion.cpp)

# Add the sdkit binary
file(GLOB_RECURSE ED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(sdkit ${ED_SRC})

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp)

# Suppress Unicode character warnings from Crow library headers on Windows
if(MSVC)
    target_compile_options(sdkit PRIVATE /wd4566)
endif()

# Find Threads package for multithreading support
find_package(Threads REQUIRED)

# Set RPATH for Linux/Unix to find shared libraries in the same directory as the executable
if(UNIX AND NOT APPLE)
    set_target_properties(sdkit PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# For macOS, use @executable_path
if(APPLE)
    set_target_properties(sdkit PROPERTIES
        BUILD_RPATH "@executable_path"
        INSTALL_RPATH "@executable_path"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

target_link_libraries(sdkit PRIVATE ggml)
target_link_libraries(sdkit PRIVATE stable-diffusion)
target_link_libraries(sdkit PRIVATE Threads::Threads)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(sdkit PRIVATE ws2_32)
    # Target Windows 7 and later (0x0601). This sets the _WIN32_WINNT macro
    # for all targets processed by this CMakeLists.
    add_compile_definitions(_WIN32_WINNT=0x0601)
endif()

# Copy the necessary DLLs to the runtime output directory
add_custom_command(TARGET sdkit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:stable-diffusion>
        $<TARGET_FILE:ggml>
        $<TARGET_FILE_DIR:sdkit>
)

# Build tools if enabled
if(SDKIT_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
