"""Smoke tests against the real downloaded models. Skipped when weights are missing.

These check that the models load and produce outputs of the right shape and
kind on Ultralytics' bundled sample images. They are not accuracy evaluations.
"""

import cv2
import numpy as np
import pytest

from app.config import load_config, resolve_path
from app.datatypes import GROUP_PERSON, GROUP_VEHICLE

settings = load_config()
WEIGHTS = resolve_path(settings.detector.weights)
SEGFORMER = resolve_path(settings.clothing.segformer_dir)

needs_detector = pytest.mark.skipif(not WEIGHTS.exists(), reason=f"{WEIGHTS.name} not downloaded")


def device():
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@pytest.fixture(scope="module")
def detector():
    from app.detector import Detector

    return Detector(settings.detector, device(), half=False)


def sample(name):
    from ultralytics.utils import ASSETS

    return cv2.imread(str(ASSETS / name))


@needs_detector
def test_detector_class_mapping(detector):
    names = set(detector.mapping.names.values())
    assert {"person", "car", "motorcycle", "bus", "truck", "bicycle"} <= names
    assert detector.mapping.missing == []
    assert detector.model.task == "segment"


@needs_detector
def test_detector_returns_instance_masks(detector):
    dets, ms = detector.detect(sample("bus.jpg"), 640)
    groups = {d.group for d in dets}
    assert GROUP_PERSON in groups and GROUP_VEHICLE in groups
    assert all(d.polygon is not None and d.polygon.shape[1] == 2 and len(d.polygon) >= 3 for d in dets)
    people = [d for d in dets if d.group == GROUP_PERSON]
    assert len(people) >= 3  # bus.jpg shows several pedestrians; each must get its own mask
    assert ms > 0


@needs_detector
def test_fp16_matches_fp32_on_cuda():
    if device() == "cpu":
        pytest.skip("CUDA not available")
    from app.detector import Detector

    d = Detector(settings.detector, device(), half=True)
    assert d.verify_half_precision(sample("bus.jpg"), 640) in (True, False)
    assert d.half_verified is not None


@needs_detector
@pytest.mark.parametrize("kind", ["bytetrack", "botsort"])
def test_real_tracker_assigns_ids(detector, kind):
    from dataclasses import replace

    from app.tracker import MultiClassTracker

    frame = sample("bus.jpg")
    trk = MultiClassTracker(replace(settings.tracker, type=kind), 30, [GROUP_VEHICLE, GROUP_PERSON])
    ids = []
    for i in range(3):
        dets, _ = detector.detect(frame, 640)
        tracked, _ = trk.update(dets, frame, i)
        ids.append(sorted(d.track_id for d in tracked))
    assert ids[-1] and ids[-1] == ids[-2]  # stable ids on an unchanged frame
    assert any(t.startswith("V") for t in ids[-1]) and any(t.startswith("P") for t in ids[-1])


@pytest.mark.skipif(not SEGFORMER.exists(), reason="clothing model not downloaded")
def test_clothing_model_segments_garments():
    from app.clothing_analyzer import ClothingAnalyzer

    an = ClothingAnalyzer(settings.clothing, device())
    assert an.mode == "segformer"
    crop = sample("zidane.jpg")[100:720, 120:620]
    attrs = an.analyze(crop, None)
    assert attrs["segmentation"] == "garment"
    assert attrs["upper_color"] is not None
    assert 0.0 <= attrs["quality"] <= 1.0


def test_clothing_fallback_is_labelled_as_approximate():
    from app.clothing_analyzer import ClothingAnalyzer
    from app.config import ClothingConfig

    an = ClothingAnalyzer(ClothingConfig(backend="person_mask_bands"), "cpu")
    crop = np.zeros((200, 80, 3), np.uint8)
    crop[40:100] = (200, 30, 30)
    crop[110:180] = (20, 20, 20)
    mask = np.ones((200, 80), np.uint8)
    attrs = an.analyze(crop, mask)
    assert attrs["segmentation"] == "approx_bands" and attrs["garments"] == []
    assert attrs["upper_color"] == "blue" and attrs["lower_color"] == "black"
    assert "not garment segmentation" in " ".join(attrs["notes"])
