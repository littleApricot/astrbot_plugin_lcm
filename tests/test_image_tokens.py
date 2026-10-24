"""Check model-specific estimates against the official image calculator."""

import base64
import io

import pytest
from PIL import Image

from astrbot_plugin_lcm.image_context import ImageContext
from astrbot_plugin_lcm.image_tokens import ImageTokenEstimator, deepseek_image_tokens
from astrbot_plugin_lcm.tests import test_integration
from astrbot_plugin_lcm.tests.test_image_context import image_count, message

owner = test_integration.owner


def image(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize(
    "width,height,tokens",
    [
        (1, 1, 184),
        (256, 256, 184),
        (512, 512, 184),
        (544, 544, 184),
        (640, 480, 206),
        (1024, 1024, 652),
        (1300, 1300, 994),
        (2000, 2000, 994),
        (5000, 5000, 994),
        (1920, 1080, 968),
        (1080, 1920, 986),
        (8192, 1, 1024),
        (1, 8192, 1024),
    ],
)
def test_official_v41_calculator_vectors(width, height, tokens):
    # Outputs from the official calculator's V4.1 implementation on 2026-10-24.
    assert deepseek_image_tokens(width, height) == tokens


def test_read_base64_and_local_headers_and_detail(tmp_path):
    reference = image(1024, 1024)
    estimator = ImageTokenEstimator("deepseek-flash")
    part = message("", reference=reference)["content"][1]
    assert estimator.count(part) == 652
    part["image_url"]["detail"] = "low"
    assert estimator.count(part) == 184
    path = tmp_path / "image with spaces.png"
    path.write_bytes(base64.b64decode(reference.split(",", 1)[1]))
    for ref in (str(path), path.as_uri()):
        assert estimator.count({"image_url": {"url": ref}}) == 652


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-flash",
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
        "deepseek/deepseek-flash",
    ],
)
def test_unavailable_dimensions_use_deepseek_upper_bound_without_network(model):
    estimator = ImageTokenEstimator(model)
    for ref in (
        "https://unreachable.invalid/image.png",
        "/missing/image.png",
        "data:image/png;base64,invalid",
    ):
        assert estimator.count({"image_url": {"url": ref}}) == 1024
    assert estimator.method == "deepseek-v4.1-dimensions"


def test_other_models_keep_explicit_generic_estimate():
    estimator = ImageTokenEstimator("unknown-model")
    assert estimator.count({"image_url": {"url": image(1024, 1024)}}) == 765
    assert estimator.method == "astrbot-generic"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,expected,used",
    [
        (0, 0, 0),
        (183, 0, 0),
        (184, 1, 184),
        (835, 1, 184),
        (836, 2, 836),
        (1020, 3, 1020),
    ],
)
async def test_token_budget_keeps_recent_suffix_and_exact_boundary(
    owner, budget, expected, used
):
    originals = [
        message("old small", reference=image(256, 256)),
        message("large", reference=image(1024, 1024)),
        message("recent small", reference=image(256, 256)),
    ]
    projection = ImageContext(owner)
    # A zero count limit is irrelevant in token mode.
    result = await projection.history(
        originals, 0, token_budget=budget, model="deepseek-flash"
    )
    assert image_count(result) == expected
    assert projection.stats["estimated_tokens"] == used <= budget
    assert projection.stats["retained_images"] == expected
    assert image_count(originals) == 3
    assert [item["content"][1]["type"] for item in result] == ["text"] * (
        3 - expected
    ) + ["image_url"] * expected


@pytest.mark.asyncio
async def test_newest_image_over_budget_does_not_skip_to_cheaper_old_image(owner):
    result = await ImageContext(owner).history(
        [
            message("old small", reference=image(256, 256)),
            message("newest large", reference=image(1024, 1024)),
        ],
        5,
        token_budget=200,
        model="deepseek-flash",
    )
    assert image_count(result) == 0


@pytest.mark.asyncio
async def test_count_mode_stays_independent_from_token_budget(owner):
    projection = ImageContext(owner)
    result = await projection.history([message("a", 3)], 2, model="deepseek-flash")
    assert image_count(result) == 2
    assert projection.stats["mode"] == "count"
