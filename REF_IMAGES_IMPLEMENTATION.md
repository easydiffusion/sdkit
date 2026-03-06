# Reference Images Implementation Guide

## Current Status

### `init_image` Implementation (Already Complete)

The `init_image` feature is fully implemented across the stack:

#### 1. **Frontend/API Layer** - [image_generator.h](include/image_generator.h#L34)
```cpp
struct ImageGenerationParams {
    std::string init_image_base64;  // Base64 encoded image
    float strength = 0.75f;         // Strength of the init image
};
```

#### 2. **Backend Processing** - [image_generator.cpp](src/image_generator.cpp#L390-L410)
- **Lines 390-394**: Decode and process init_image
  - Checks if img2img mode is enabled and init_image_base64 is provided
  - Calls `createInitImage()` to decode and resize the image
  
- **Lines 499-509**: `createInitImage()` function
  - Decodes base64 to image using `base64ToImage()`
  - Resizes image to match width/height parameters
  - Returns sd_image_t struct

- **Lines 449-451**: Cleanup
  - Frees the init_image memory after generation

#### 3. **Core Engine** - [stable-diffusion.h](stable-diffusion.cpp/stable-diffusion.h#L287-L306)
```cpp
typedef struct {
    sd_image_t init_image;      // The initial image for img2img
    sd_image_t* ref_images;     // Reference images (NOT YET EXPOSED TO FRONTEND)
    int ref_images_count;       // Number of reference images
    bool auto_resize_ref_image; // Auto-resize setting
    // ... other fields
} sd_img_gen_params_t;
```

### `ref_images` Implementation (Backend Only)

The `ref_images` feature **already exists in the backend** but is NOT exposed through the ImageGenerator layer:

#### 1. **Core Engine Support** - [stable-diffusion.h](stable-diffusion.cpp/stable-diffusion.h#L289-L291)
- `sd_image_t* ref_images` - array of reference images
- `int ref_images_count` - count of images
- `bool auto_resize_ref_image` - auto-resize flag

#### 2. **Usage in Backend** - [stable-diffusion.cpp](stable-diffusion.cpp/stable-diffusion.cpp#L3730)
- Line 3730: `if (sd_img_gen_params->init_image.data != nullptr || sd_img_gen_params->ref_images_count > 0)`
- Lines 3662-3664: Reference images are pushed into a vector for processing
- Line 3669: Empty image is used if no ref_images for certain models
- Lines 1753-1764: [conditioner.hpp](stable-diffusion.cpp/conditioner.hpp#L1753-L1764) - Vision processing of ref_images for Qwen image editing

---

## What's Missing

To fully implement `ref_images` support, you need to add to the **frontend layer** ([image_generator.h](include/image_generator.h) and [image_generator.cpp](src/image_generator.cpp)):

### Changes Needed:

1. **Add to ImageGenerationParams struct** ([image_generator.h](include/image_generator.h#L17))
   ```cpp
   struct ImageGenerationParams {
       // ... existing fields ...
       
       // Reference images for Qwen/vision-based models
       std::vector<std::string> ref_images_base64;  // Base64 encoded reference images
       bool auto_resize_ref_image = true;
   };
   ```

2. **Add helper function in ImageGenerator class** ([image_generator.cpp](src/image_generator.cpp))
   ```cpp
   // Similar to createInitImage() - create function to process ref_images
   // Decode each base64 string, resize, and return vector of sd_image_t
   ```

3. **Process ref_images in generateImage()** (around line 390-410)
   ```cpp
   // Allocate array for reference images
   // Loop through params.ref_images_base64
   // Decode and resize each one
   // Set gen_params.ref_images and gen_params.ref_images_count
   ```

4. **Cleanup ref_images memory** (around line 449-451)
   ```cpp
   // Free each reference image after generation
   ```

---

## File Locations Summary

| Component | File | Key Lines |
|-----------|------|-----------|
| API Definition | [include/image_generator.h](include/image_generator.h) | 17-45 (struct) |
| Frontend Logic | [src/image_generator.cpp](src/image_generator.cpp) | 390-410, 499-509, 449-451 |
| Backend Struct | [stable-diffusion.cpp/stable-diffusion.h](stable-diffusion.cpp/stable-diffusion.h) | 287-306 |
| Backend Usage | [stable-diffusion.cpp/stable-diffusion.cpp](stable-diffusion.cpp/stable-diffusion.cpp) | 3730, 3662-3664, 3669 |
| Vision Processing | [stable-diffusion.cpp/conditioner.hpp](stable-diffusion.cpp/conditioner.hpp) | 1753-1764 |
