"""Known CRA373 joint limits, with explicit unknown physical capabilities."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

@lru_cache(maxsize=1)
def read_robot_capability() -> dict:
    urdf = Path(__file__).resolve().parents[2] / "CRA373_model" / "cra373.urdf"
    root = ET.parse(urdf).getroot()
    limits = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        bound = joint.find("limit")
        if bound is not None:
            limits[joint.attrib["name"]] = {
                "lower": float(bound.attrib["lower"]),
                "upper": float(bound.attrib["upper"]),
            }
    return {
        "joint_limits_rad": limits,
        "body_dimensions_m": None,
        "leg_reach_m": None,
        "validated_max_body_tilt_deg": None,
        "validated_step_height_m": None,
        "validated_jump_capability": False,
        "source": str(urdf),
    }
