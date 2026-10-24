from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from transformers import CLIPVisionConfig

from deepfake_core.detectors.gend import (
    FAKE_INDEX,
    GenDDetector,
    GenDNetwork,
    preprocess,
    remap_checkpoint,
)
from deepfake_core.errors import ModelIntegrityError, ModelMissingError
from tools.fetch_fixtures import FIXTURES_DIR, GEND_SAMPLES

REFERENCE = json.loads((Path(__file__).parent / "data" / "gend_reference.json").read_text())
TINY = CLIPVisionConfig(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    image_size=224,
    patch_size=56,
    hidden_act="quick_gelu",
    projection_dim=16,
)


def gend_checkpoint_from(network: GenDNetwork) -> dict[str, torch.Tensor]:
    """A checkpoint with GenD's on-disk key layout, built from `network`'s weights."""
    checkpoint: dict[str, torch.Tensor] = {}
    for key, tensor in network.vision.state_dict().items():
        if key.endswith("position_ids"):
            continue
        checkpoint["feature_extractor.vision_model." + key.removeprefix("vision_model.")] = tensor
    checkpoint["feature_extractor.visual_projection.weight"] = torch.zeros(16, 32)
    checkpoint["model.linear.weight"] = network.head.weight.detach().clone()
    checkpoint["model.linear.bias"] = network.head.bias.detach().clone()
    return {k: v.contiguous() for k, v in checkpoint.items()}


def test_tiny_checkpoint_round_trip_and_linear_norm_head(tmp_path: Path) -> None:
    torch.manual_seed(0)
    source = GenDNetwork(TINY).eval()
    path = tmp_path / "model.safetensors"
    save_file(gend_checkpoint_from(source), path)

    detector = GenDDetector(checkpoint=path, network_config=TINY, device="cpu", batch_size=3)
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(5)]
    result = detector.predict(crops)

    assert result.p_fake.shape == (5,) and result.embeddings.shape == (5, 32)
    np.testing.assert_allclose(np.linalg.norm(result.embeddings, axis=1), 1.0, atol=1e-5)
    with torch.inference_mode():
        pooled = source.vision(pixel_values=preprocess(crops)).pooler_output
        expected = source.head(F.normalize(pooled, dim=1)).softmax(-1)[:, FAKE_INDEX]
    np.testing.assert_allclose(result.p_fake, expected.numpy(), atol=1e-5)


def test_remap_rejects_unexpected_and_missing_tensors() -> None:
    network = GenDNetwork(TINY)
    checkpoint = gend_checkpoint_from(network)
    with pytest.raises(ModelIntegrityError, match="Unexpected tensor"):
        remap_checkpoint({**checkpoint, "model.extra": torch.zeros(1)}, network)
    del checkpoint["model.linear.bias"]
    with pytest.raises(ModelIntegrityError, match="missing"):
        remap_checkpoint(checkpoint, network)


def test_preprocess_matches_clip_normalisation() -> None:
    crop = np.full((224, 224, 3), 128, np.uint8)
    pixels = preprocess([crop])
    assert pixels.shape == (1, 3, 224, 224)
    expected = (128 / 255 - 0.48145466) / 0.26862954
    assert float(pixels[0, 0, 0, 0]) == pytest.approx(expected, abs=1e-6)


@pytest.fixture(scope="module")
def gend() -> Any:
    try:
        return GenDDetector()
    except ModelMissingError:
        pytest.skip("GenD weights missing: run `make models`")


@pytest.mark.models
def test_matches_upstream_gend_on_its_sample_crops(gend: Any) -> None:
    """Pins preprocessing, the LinearNorm head and the label index against upstream GenD."""
    paths = [FIXTURES_DIR / fixture.path for fixture in GEND_SAMPLES]
    if not all(path.is_file() for path in paths):
        pytest.skip("GenD sample crops missing: run `make fixtures`")
    crops = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    result = gend.predict(crops)
    for fixture, ours in zip(GEND_SAMPLES, result.p_fake, strict=True):
        reference = REFERENCE["samples"][fixture.path.removeprefix("gend_samples/")]
        assert abs(float(ours) - reference["p_fake"]) < 0.01, fixture.path
    np.testing.assert_allclose(np.linalg.norm(result.embeddings, axis=1), 1.0, atol=1e-4)
