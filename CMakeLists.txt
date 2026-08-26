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

# Common libraries in third_party:
#  - ggml: a single ggml, shared by stable-diffusion.cpp and sam3.cpp.
#
#    Both libraries normally vendor their own ggml copy as a submodule. To keep
#    a single ggml (and a single shared ggml artifact), we build ggml from
#    third_party/ggml first:
#      - stable-diffusion.cpp's CMakeLists skips its vendored ggml when a `ggml`
#        target already exists (`if(NOT TARGET ggml)`).
#      - sam3.cpp's CMakeLists unconditionally adds its own ggml, so we don't
#        include it; instead we build its single-file library directly below.
#
#    Backend selection (CUDA / Metal / Vulkan / CPU flags etc.) is done by
#    passing GGML_* variables directly on the CMake command line — both sd.cpp
#    and sam3 work with any backend that the shared ggml was built with.
#
# stable-diffusion.cpp expects this macro (it normally defines it right before
# adding its vendored ggml). It must be set before ggml is added here so that
# all consumers share the same ABI.
add_compile_definitions(GGML_MAX_NAME=128)

add_subdirectory(third_party/ggml)

# Common stb (header-only image loading/writing), shared by the sdkit binary and
# sam3.cpp. Both compile stb implementations into their own binaries; each does
# so with static symbols (or is the only non-static user) to avoid link clashes.
# stable-diffusion.cpp keeps using its own vendored stb headers internally; its
# implementations are static or absent, so they can't clash with ours.
set(STB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

# sam3.cpp: image/video segmentation library (SAM 1/2/3, EdgeTAM). Built as a
# static lib from its single-file sources, linked against the common ggml.
add_library(sam3 STATIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sam3.cpp/sam3.cpp)
target_include_directories(sam3 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sam3.cpp
    ${STB_INCLUDE_DIR}
)
target_link_libraries(sam3 PUBLIC ggml)
# sam3.cpp compiles its own copies of stb_image/stb_image_write. Make those
# symbols static so they don't clash with the stb symbols compiled into the
# sdkit binary itself.
target_compile_definitions(sam3 PRIVATE STB_IMAGE_STATIC STB_IMAGE_WRITE_STATIC)

# Include stable-diffusion.cpp
add_subdirectory(third_party/stable-diffusion.cpp)

# Add the sdkit binary
file(GLOB_RECURSE ED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(sdkit ${ED_SRC})

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

target_include_directories(sdkit PRIVATE ${STB_INCLUDE_DIR})

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp)

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sam3.cpp)

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
target_link_libraries(sdkit PRIVATE sam3)
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
