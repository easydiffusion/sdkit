from PIL import Image

from sdkit import Context
from sdkit.generate import generate_images
from sdkit.models import load_model, unload_model
from sdkit.filter import apply_filters


from common import (
    TEST_DATA_FOLDER,
    assert_images_same,
    run_test_on_multiple_devices,
)

INPUT_DIR = f"{TEST_DATA_FOLDER}/input_images"
EXPECTED_DIR = f"{TEST_DATA_FOLDER}/expected_images"

context = None


def setup_module():
    global context

    context = Context()
    context.test_diffusers = True


# section 1 - pre-processors as filters
def image_filter_test(img, check_image=True):
    prev_filter = None

    from sdkit.models.model_loader.controlnet_filters import filters

    for filter_name in filters:
        if prev_filter:
            unload_model(context, prev_filter)

        context.model_paths[filter_name] = filter_name
        load_model(context, filter_name)

        filtered_img = apply_filters(context, filter_name, img)
        assert filtered_img is not None
        assert len(filtered_img) > 0, f"{filter_name} length {len(filtered_img)} > 0"
        assert None not in filtered_img, f"{filter_name} {filtered_img} has None"

        if check_image and filter_name not in ("shuffle", "segment"):  # shuffle and segment are random
            filtered_img = filtered_img[0]
            expected_image = Image.open(f"{EXPECTED_DIR}/filters/{filter_name}.png")
            assert_images_same(filtered_img, expected_image, "test_" + filter_name)

        prev_filter = filter_name


def test_1_1__can_preprocess_single_image_via_filter():
    img = Image.open(f"{INPUT_DIR}/pose.jpg")
    image_filter_test(img)


def test_1_2__can_preprocess_multiple_images_via_filter():
    img = [
        Image.open(f"{INPUT_DIR}/pose.jpg"),
        Image.open(f"{INPUT_DIR}/painting.jpg"),
    ]
    image_filter_test(img, check_image=False)


# section 2 - load/unload controlnet
def test_load_sd_model():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"
    load_model(context, "stable-diffusion")


def test_2_1__load_controlnet():
    context.model_paths["controlnet"] = "models/controlnet/control_v11p_sd15_openpose.pth"
    load_model(context, "controlnet")

    assert context.models["controlnet"] is not None


def test_2_2__load_multiple_controlnets():
    context.model_paths["controlnet"] = [
        "models/controlnet/control_v11p_sd15_canny.pth",
        "models/controlnet/control_v11p_sd15_openpose.pth",
    ]
    load_model(context, "controlnet")

    assert context.models["controlnet"] is not None


def test_2_3__unload_controlnet():
    unload_model(context, "controlnet")

    assert "controlnet" not in context.models


# section 3 - no difference if controlnet isn't set properly
def test_3_1a__no_difference_if_control_image_is_not_passed__txt2img():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=512,
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-no_controlnet.png")
    assert_images_same(image, expected_image, "test3.1a")


def test_3_1b__no_difference_if_control_image_is_not_passed__img2img():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=512,
        init_image=Image.open(f"{EXPECTED_DIR}/controlnet/txt-no_controlnet.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/img-no_controlnet.png")
    assert_images_same(image, expected_image, "test3.1b")


def test_3_1c__no_difference_if_control_image_is_not_passed__inpainting():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=512,
        init_image=Image.open(f"{INPUT_DIR}/dog-512x512.png"),
        init_image_mask=Image.open(f"{INPUT_DIR}/dog_mask-512x512.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/inpaint-no_controlnet.png")
    assert_images_same(image, expected_image, "test3.1c")


def test_3_2__no_difference_if_control_image_is_passed_but_controlnet_model_not_loaded():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=512,
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-no_controlnet.png")
    assert_images_same(image, expected_image, "test3.2")


def test_3_3__no_difference_if_control_image_is_not_passed_but_controlnet_model_is_loaded():
    test_2_1__load_controlnet()

    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=512,
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-no_controlnet.png")
    assert_images_same(image, expected_image, "test3.3")


# section 4 - controlnet images
def test_load_openpose_controlnet():
    context.model_paths["controlnet"] = "models/controlnet/control_v11p_sd15_openpose.pth"
    load_model(context, "controlnet")


def test_4_1a__generates_image_from_single_control_image__txt2img():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-pose_controlnet.png")
    assert_images_same(image, expected_image, "test4.1a")


def test_4_1b__generates_image_from_single_control_image__img2img():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        init_image=Image.open(f"{INPUT_DIR}/pose.jpg"),
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/img-pose_controlnet.png")
    assert_images_same(image, expected_image, "test4.1b")


def test_4_1c__generates_image_from_single_control_image__inpaint():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        init_image=Image.open(f"{INPUT_DIR}/pose.jpg"),
        init_image_mask=Image.open(f"{INPUT_DIR}/pose_mask.png"),
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/inpaint-pose_controlnet.png")
    assert_images_same(image, expected_image, "test4.1c")


def test_4_1d1__generates_image_from_single_control_image_with_zero_strength():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
        control_alpha=0,
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-pose_controlnet_alpha_0.png")
    assert_images_same(image, expected_image, "test4.1d1")


def test_4_1d2__generates_image_from_single_control_image_with_mid_strength():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
        control_alpha=0.6,
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-pose_controlnet_alpha_0.6.png")
    assert_images_same(image, expected_image, "test4.1d2")


def test_4_1d3__generates_image_from_single_control_image_with_full_strength():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
        control_alpha=1,
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-pose_controlnet.png")
    assert_images_same(image, expected_image, "test4.1d3")


def test_load_openpose_and_canny_controlnets():
    context.model_paths["controlnet"] = [
        "models/controlnet/control_v11p_sd15_openpose.pth",
        "models/controlnet/control_v11p_sd15_canny.pth",
    ]
    load_model(context, "controlnet")


def test_4_2a__generates_image_from_multiple_control_images__txt2img():
    image = generate_images(
        context,
        "1boy, standing in a field of trees, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        control_image=[
            Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
            Image.open(f"{INPUT_DIR}/canny_trees_masked.png"),
        ],
        control_alpha=[
            1.0,
            0.8,
        ],
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-pose_canny_controlnet.png")
    assert_images_same(image, expected_image, "test4.2a")


def test_4_2b__generates_image_from_multiple_control_images__img2img():
    image = generate_images(
        context,
        "1boy, standing in a field of trees, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        init_image=Image.open(f"{INPUT_DIR}/pose.jpg"),
        control_image=[
            Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
            Image.open(f"{INPUT_DIR}/canny_trees_masked.png"),
        ],
        control_alpha=[
            1.0,
            0.8,
        ],
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/img-pose_canny_controlnet.png")
    assert_images_same(image, expected_image, "test4.2b")


def test_4_2c__generates_image_from_multiple_control_images__img2img():
    image = generate_images(
        context,
        "1boy, standing in a field of trees, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        init_image=Image.open(f"{INPUT_DIR}/pose.jpg"),
        init_image_mask=Image.open(f"{INPUT_DIR}/pose_mask.png"),
        control_image=[
            Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
            Image.open(f"{INPUT_DIR}/canny_trees_masked.png"),
        ],
        control_alpha=[
            1.0,
            0.8,
        ],
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/inpaint-pose_canny_controlnet.png")
    assert_images_same(image, expected_image, "test4.2c")


# section 5 - multiple devices
def run_on_devices(vram_usage_level):
    def task(context):
        context.test_diffusers = True
        context.vram_usage_level = vram_usage_level

        # load SD
        context.model_paths["stable-diffusion"] = "models/stable-diffusion/sd-v1-4.ckpt"
        load_model(context, "stable-diffusion")

        # load controlnet
        context.model_paths["controlnet"] = "models/controlnet/control_v11p_sd15_openpose.pth"
        load_model(context, "controlnet")

        image = generate_images(
            context,
            "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
            seed=42,
            width=512,
            height=768,
            num_inference_steps=1,
            control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
        )[0]

        assert image is not None, f"{context.device} {vram_usage_level} - image is None!"
        assert image.getbbox(), f"{context.device} {vram_usage_level} - image is black!"

    run_test_on_multiple_devices(task=task, devices=["cuda:0", "cpu"])


def test_5_1a__generates_image_from_single_control_image__low():
    run_on_devices(vram_usage_level="low")


def test_5_1b__generates_image_from_single_control_image__balanced():
    run_on_devices(vram_usage_level="balanced")


def test_5_1c__generates_image_from_single_control_image__high():
    run_on_devices(vram_usage_level="high")


# section 6 - SD-XL
def test_load_sdxl_model():
    context.model_paths["stable-diffusion"] = "models/stable-diffusion/official/sd_xl_base_1.0.safetensors"
    load_model(context, "stable-diffusion")


def test_6_1__load_controlnet():
    context.model_paths["controlnet"] = "models/controlnet/OpenPoseXL2.safetensors"
    load_model(context, "controlnet")

    assert context.models["controlnet"] is not None


def test_6_2__generates_image_from_single_control_image__txt2img():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/txt-pose_controlnet_xl.png")
    assert_images_same(image, expected_image, "test6.2")


def test_6_3__generates_image_from_single_control_image__img2img():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        init_image=Image.open(f"{INPUT_DIR}/pose.jpg"),
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/img-pose_controlnet_xl.png")
    assert_images_same(image, expected_image, "test6.3")


def test_6_4__generates_image_from_single_control_image__inpaint():
    image = generate_images(
        context,
        "1boy, muscular, full armor, armor, shoulder plates, angry, looking at viewer, spiked hair, white hair",
        seed=42,
        width=512,
        height=768,
        init_image=Image.open(f"{INPUT_DIR}/pose.jpg"),
        init_image_mask=Image.open(f"{INPUT_DIR}/pose_mask.png"),
        control_image=Image.open(f"{EXPECTED_DIR}/filters/openpose.png"),
    )[0]

    expected_image = Image.open(f"{EXPECTED_DIR}/controlnet/inpaint-pose_controlnet_xl.png")
    assert_images_same(image, expected_image, "test6.4")
