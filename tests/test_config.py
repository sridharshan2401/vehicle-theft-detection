import pytest

from app.config import DEFAULT_CONFIG_PATH, ConfigError, Settings, load_config, validate


def test_default_settings_file_loads():
    s = load_config()
    assert DEFAULT_CONFIG_PATH.exists()
    assert s.app.mode in ("responsive", "complete")
    assert s.detector.weights.endswith("-seg.pt")
    assert s.privacy.face_processing is False


def test_builtin_defaults_are_valid():
    validate(Settings())


def test_dotted_overrides_apply():
    s = load_config(None, {"app.mode": "complete", "detector.imgsz": 512, "risk.levels.LOW": 10})
    assert s.app.mode == "complete"
    assert s.detector.imgsz == 512
    # partial dict override keeps the other levels
    assert s.risk.levels["LOW"] == 10 and s.risk.levels["CRITICAL"] == 80


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("detector:\n  weigths: x.pt\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(p)


def test_wrong_type_rejected():
    with pytest.raises(ConfigError, match="expected a number"):
        load_config(None, {"detector.conf": "high"})
    with pytest.raises(ConfigError, match="expected true/false"):
        load_config(None, {"app.display": "yes"})


def test_invalid_values_rejected():
    with pytest.raises(ConfigError, match="app.mode"):
        load_config(None, {"app.mode": "fast"})
    with pytest.raises(ConfigError, match="multiple of 32"):
        load_config(None, {"detector.imgsz": 650})
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(None, {"risk.levels": {"LOW": 50, "MEDIUM": 40, "HIGH": 60, "CRITICAL": 80}})
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_config(None, {"association.weights.door_zone": 0.9})


def test_missing_and_malformed_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "broken.yaml"
    bad.write_text("app: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Could not parse"):
        load_config(bad)
