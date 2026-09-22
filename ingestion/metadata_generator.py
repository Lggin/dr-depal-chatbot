"""Chunk metadata (spec §6) from per-source config + rule-based extraction. No LLM."""

import re
from typing import Any, Dict, List, Optional

from app.router.alarm_parser import extract_alarm_codes
from ingestion.chunker import Chunk

# keyword (lowercase) -> component name. Systems are inferred from components.
COMPONENT_KEYWORDS: Dict[str, str] = {
    "rf-detr": "rf-detr", "rfdetr": "rf-detr", "rf_detr": "rf-detr",
    "sam": "sam", "lora": "lora", "onnx": "onnx", "tensorrt": "tensorrt", "trt": "tensorrt",
    "ransac": "ransac", "kalman": "kalman", "imu": "imu",
    "dr-depal-sdk": "dr-depal-sdk", "dr-camera-sdk": "dr-camera-sdk", "dr-config-sdk": "dr-config-sdk",
    "dr-logging-sdk": "dr-logging-sdk", "dr-drift-monitor-sdk": "dr-drift-monitor-sdk",
    "dr-obj-und-sdk": "dr-obj-und-sdk", "dr-point-cloud-sdk": "dr-point-cloud-sdk",
    "dr-calibration-sdk": "dr-calibration-sdk", "onsite-adap": "dr-onsite-adap-sdk",
    "femto": "femto-mega", "orbbec": "femto-mega", "jetson": "jetson", "orin": "jetson", "thor": "thor",
    "nexus": "nexus", "lakefs": "lakefs", "replay": "replay", "quantiz": "quantizer",
}
COMPONENT_SYSTEM: Dict[str, str] = {
    "rf-detr": "vision", "sam": "vision", "dr-obj-und-sdk": "vision", "dr-camera-sdk": "vision", "femto-mega": "vision",
    "dr-calibration-sdk": "vision", "ransac": "geometry", "kalman": "geometry", "imu": "geometry", "dr-point-cloud-sdk": "geometry",
    "lora": "training", "replay": "training", "quantizer": "training", "dr-onsite-adap-sdk": "onsite_adaptation",
    "onnx": "deployment", "tensorrt": "deployment", "jetson": "deployment", "thor": "deployment", "nexus": "release", "lakefs": "release",
    "dr-config-sdk": "config", "dr-depal-sdk": "runtime", "dr-logging-sdk": "runtime", "dr-drift-monitor-sdk": "runtime",
}

_VERSION = re.compile(r"\bv?(0\.0\.\d{1,3})\b")
_ISSUE = re.compile(r"(?<![\w/])#(\d{2,5})\b")
_DATE = re.compile(r"\b(20\d{2})[-./](\d{1,2})[-./](\d{1,2})\b")
_NUMERIC = re.compile(r"\d+(?:\.\d+)?\s*(?:ms|s|초|mm|cm|m|kg|kPa|MPa|fps|FPS|%|GB|MB|px|°|도)\b")


def detect_components(text: str) -> List[str]:
    low = text.lower()
    found: List[str] = []
    for kw, comp in COMPONENT_KEYWORDS.items():
        if kw in low and comp not in found:
            found.append(comp)
    return found


def infer_system(components: List[str], default: Optional[str]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for c in components:
        s = COMPONENT_SYSTEM.get(c)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return default
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] >= 2 or default is None or default == "general" else default


def latest_date(text: str) -> Optional[str]:
    dates = [f"{y}-{int(m):02d}-{int(d):02d}" for y, m, d in _DATE.findall(text) if 1 <= int(m) <= 12 and 1 <= int(d) <= 31]
    return max(dates) if dates else None


def build_metadata(chunk: Chunk, source_cfg: Dict[str, Any], globals_cfg: Dict[str, Any]) -> Dict[str, Any]:
    text = chunk.search_text
    components = detect_components(text)
    versions = sorted({f"v{v}" for v in _VERSION.findall(text)})
    issues = sorted({f"#{i}" for i in _ISSUE.findall(text)})
    tags = list(dict.fromkeys(list(source_cfg.get("tags", [])) + versions[:5] + issues[:5]))

    meta: Dict[str, Any] = {
        "machine": globals_cfg.get("machine"),
        "manufacturer": globals_cfg.get("manufacturer"),
        "document_type": source_cfg["document_type"],
        "document_name": source_cfg.get("document_name") or chunk.document_title or chunk.source,
        "source": chunk.source,
        "section": chunk.section,
        "heading_path": chunk.heading_path,
        "system": infer_system(components, source_cfg.get("system")),
        "component": components[0] if components else None,
        "components": components,
        "alarm_code": None,
        "symptoms": [],
        "software_version": versions[-1] if versions else None,
        "language": globals_cfg.get("language", "ko"),
        "trust_level": int(source_cfg["trust_level"]),
        "tags": tags,
        "as_of_date": latest_date(text),
        "contains_numeric_spec": bool(_NUMERIC.search(chunk.text)),
        "part": chunk.part,
        "n_parts": chunk.n_parts,
        "tokens": chunk.tokens,
        "search_text": chunk.search_text,
    }
    if globals_cfg.get("alarm_codes", True):
        codes = extract_alarm_codes(chunk.text)
        meta["alarm_code"] = codes[0] if codes else None
    if chunk.merged_from:
        meta["merged_from"] = chunk.merged_from
    return meta
