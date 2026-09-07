"""Render PATH translation and bottle-normal orientation tracking from trial_013."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


SOURCE = Path(r"C:\studioprojects\PATH-G1\record\controller\PATH\trial_013.csv")
OUTPUT = Path("public/data/path_reference_relation_trial_013.gif")
FRAME_COUNT = 180
FRAME_DURATION_MS = 70
HAND_APPROACH_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])
NORMAL_LPF_ALPHA = 0.03
BOTTLE_POSITION_LPF_ALPHA = 0.06
ATTACH_BLEND_SECONDS = 0.60


def unit(vector: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(vector)
    return vector / max(magnitude, 1.0e-9)


def parse_pose(value: str) -> tuple[np.ndarray, np.ndarray]:
    pose = json.loads(value)
    return np.asarray(pose["position_m"], dtype=float), np.asarray(pose["orientation"], dtype=float).reshape(3, 3)


def relation_error_degrees(axis: np.ndarray, normal: np.ndarray, relation: str) -> float:
    angle = np.degrees(np.arccos(np.clip(abs(float(np.dot(unit(axis), unit(normal)))), 0.0, 1.0)))
    return angle if relation == "parallel" else abs(90.0 - angle)


def orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
    first = unit(np.cross(axis, helper))
    return first, unit(np.cross(axis, first))


def rotation_with_approach_axis(approach: np.ndarray) -> np.ndarray:
    """Create a stable Dex1 frame whose local +Y follows the given approach."""
    spread, lift = orthogonal_basis(unit(approach))
    return np.column_stack((spread, unit(approach), lift))


def roll_gripper_90(rotation: np.ndarray) -> np.ndarray:
    """Rotate a hand frame +90 degrees about its local +Y approach axis."""
    roll = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    return rotation @ roll


def draw_bottle(axis, center: np.ndarray, normal: np.ndarray) -> None:
    """A small wireframe bottle, oriented by the selected bottle normal."""
    tangent_a, tangent_b = orthogonal_basis(normal)
    height, radius = 0.16, 0.033
    base, shoulder = center - normal * height * 0.48, center + normal * height * 0.33
    cap_base, cap_top = center + normal * height * 0.40, center + normal * height * 0.53
    angles = np.linspace(0, 2 * np.pi, 18)

    def ring(point: np.ndarray, ring_radius: float, color: str, width: float) -> None:
        coordinates = np.array([point + ring_radius * (np.cos(angle) * tangent_a + np.sin(angle) * tangent_b) for angle in angles])
        axis.plot(*coordinates.T, color=color, linewidth=width, alpha=0.88)

    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        direction = np.cos(angle) * tangent_a + np.sin(angle) * tangent_b
        axis.plot(*np.vstack((base + radius * direction, shoulder + radius * direction)).T, color="#64748b", linewidth=1.5, alpha=0.72)
    ring(base, radius, "#475569", 1.7)
    ring(shoulder, radius, "#475569", 1.7)
    ring(cap_base, radius * 0.56, "#0f172a", 1.8)
    ring(cap_top, radius * 0.56, "#0f172a", 1.8)
    axis.plot(*np.vstack((shoulder, cap_base)).T, color="#334155", linewidth=2.0)


def arrow(axis, origin: np.ndarray, direction: np.ndarray, color: str, width: float, length: float = 0.115) -> None:
    axis.quiver(
        *origin,
        *(unit(direction) * length),
        color=color,
        linewidth=width,
        arrow_length_ratio=0.19,
        normalize=False,
    )


def draw_dex1_gripper(axis, origin: np.ndarray, rotation: np.ndarray, grasp_state: str, engaged: bool) -> None:
    """Draw the dark, parallel-jaw silhouette of a Dex1-1 gripper."""
    approach = unit(rotation @ np.array([0.0, 1.0, 0.0]))
    spread = unit(rotation @ np.array([1.0, 0.0, 0.0]))
    lift = unit(rotation @ np.array([0.0, 0.0, 1.0]))
    if engaged:
        gap = 0.010
    elif grasp_state == "closing":
        gap = 0.034
    else:
        gap = 0.074

    # Keep the Dex1-1 jaws aimed with the blue approach axis, but shift the
    # whole silhouette in the opposite transverse direction so it stays legible.
    visual_origin = origin - lift * 0.035
    palm = visual_origin - approach * 0.030
    jaw_start = visual_origin + approach * 0.002
    jaw_length = 0.078
    jaw_tip_inset = 0.014 if not engaged else 0.003
    color, pad_color = "#1f2937", "#94a3b8"
    axis.plot(*np.vstack((palm - spread * 0.056, palm + spread * 0.056)).T, color=color, linewidth=5.8, solid_capstyle="round")
    axis.plot(*np.vstack((palm - lift * 0.014, palm + lift * 0.014)).T, color="#0f172a", linewidth=4.8, solid_capstyle="round")
    for sign in (-1.0, 1.0):
        start = jaw_start + spread * sign * gap * 0.5
        tip = start + approach * jaw_length
        inward = tip - spread * sign * jaw_tip_inset
        axis.plot(*np.vstack((palm + spread * sign * 0.042, start, tip, inward)).T, color=color, linewidth=5.8, solid_capstyle="round")
        pad_half_length = 0.024 if engaged else 0.014
        pad_width = 5.6 if engaged else 3.4
        axis.plot(*np.vstack((inward - lift * pad_half_length, inward + lift * pad_half_length)).T, color=pad_color, linewidth=pad_width, solid_capstyle="round")


def bottle_detected(value: object) -> bool:
    if not isinstance(value, str) or value in {"", "[]", "null"}:
        return False
    try:
        objects = json.loads(value)
    except json.JSONDecodeError:
        return False
    return any(
        isinstance(item, dict)
        and "bottle" in str(item.get("label", item.get("object_id", ""))).lower()
        for item in objects
    )


def main(source: Path = SOURCE, output: Path = OUTPUT) -> None:
    rows = []
    last_observed: dict[str, np.ndarray | str] | None = None
    attachment_offset_local: np.ndarray | None = None
    attachment_normal_local: np.ndarray | None = None
    attachment_started = False
    for _, row in pd.read_csv(source).iterrows():
        raw_belief = row["left_belief_target_json"]
        raw_reference = row["left_ipopt_target_json"]
        raw_measured = row["left_ee_json"]
        raw_right_reference = row["right_ipopt_target_json"]
        raw_right_measured = row["right_ee_json"]
        if not all(isinstance(value, str) and value != "null" for value in (raw_reference, raw_measured)):
            continue
        reference_payload = json.loads(raw_reference)
        actual_position, actual_rotation = parse_pose(raw_measured)
        reference_position, reference_rotation = parse_pose(raw_reference)
        right_reference_payload = json.loads(raw_right_reference) if isinstance(raw_right_reference, str) and raw_right_reference != "null" else {}
        right_position, right_rotation = (
            parse_pose(raw_right_measured)
            if isinstance(raw_right_measured, str) and raw_right_measured != "null"
            else (None, None)
        )
        engaged = bool(reference_payload.get("grasp_engaged", False))
        detected = bottle_detected(row["left_objects_json"])
        belief = json.loads(raw_belief) if isinstance(raw_belief, str) and raw_belief != "null" else None
        if (
            detected
            and isinstance(belief, dict)
            and "bottle" in str(belief.get("object_id", ""))
            and belief.get("relation") in {"parallel", "perpendicular"}
        ):
            last_observed = {
                "position": np.asarray(belief["position_m"], dtype=float),
                "normal": unit(np.asarray(belief["orientation_axis"], dtype=float)),
                "relation": str(belief["relation"]),
            }
        if last_observed is None:
            continue

        # A true grasp_engaged signal confirms the object is in the Dex1-1
        # jaws. Switch to the hand-carried pose immediately, rather than
        # waiting for SAM3 to lose the now-occluded bottle.
        if engaged:
            if attachment_offset_local is None:
                # Once SAM3 loses the bottle after a confirmed grasp, visualize it at the
                # Dex1-1 jaw center rather than preserving a stale camera-space offset.
                attachment_offset_local = np.array([0.0, 0.105, 0.0])
                attachment_normal_local = actual_rotation.T @ last_observed["normal"]
                attachment_started = True
            bottle_position = actual_position + actual_rotation @ attachment_offset_local
            normal = unit(actual_rotation @ attachment_normal_local)
            attached = True
        else:
            bottle_position = np.asarray(last_observed["position"], dtype=float)
            normal = np.asarray(last_observed["normal"], dtype=float)
            attached = False

        if attachment_started and not engaged:
            break
        rows.append(
            {
                "time": float(row["elapsed_s"]),
                "actual_position": actual_position,
                "actual_rotation": actual_rotation,
                "right_position": right_position,
                "right_rotation": right_rotation,
                "reference_position": reference_position,
                "actual_axis": unit(actual_rotation @ HAND_APPROACH_AXIS_LOCAL),
                "reference_axis": unit(reference_rotation @ HAND_APPROACH_AXIS_LOCAL),
                "bottle_position": bottle_position,
                "normal": normal,
                "relation": str(last_observed["relation"]),
                "grasp_state": str(reference_payload.get("grasp_state", "reaching")),
                "grasp_engaged": engaged,
                "right_grasp_state": str(right_reference_payload.get("grasp_state", "reaching")),
                "right_grasp_engaged": bool(right_reference_payload.get("grasp_engaged", False)),
                "bottle_attached": attached,
            }
        )

    if not rows:
        raise RuntimeError("No bottle-oriented reference frames were found in trial_013.")

    # Surface normals are unsigned for this task. Keep a visually continuous direction.
    for index in range(1, len(rows)):
        if np.dot(rows[index - 1]["normal"], rows[index]["normal"]) < 0.0:
            rows[index]["normal"] *= -1.0

    # Filter SAM3's early normal/position jitter and smoothly hand off from the
    # last camera observation to the bottle pose carried by the gripper.
    filtered_normal = rows[0]["normal"].copy()
    filtered_position = rows[0]["bottle_position"].copy()
    attachment_start_time: float | None = None
    attachment_anchor_position: np.ndarray | None = None
    attachment_anchor_normal: np.ndarray | None = None
    for frame in rows:
        if frame["bottle_attached"]:
            if attachment_start_time is None:
                attachment_start_time = frame["time"]
                attachment_anchor_position = filtered_position.copy()
                attachment_anchor_normal = filtered_normal.copy()
            blend = float(np.clip((frame["time"] - attachment_start_time) / ATTACH_BLEND_SECONDS, 0.0, 1.0))
            filtered_position = (1.0 - blend) * attachment_anchor_position + blend * frame["bottle_position"]
            filtered_normal = unit((1.0 - blend) * attachment_anchor_normal + blend * frame["normal"])
        else:
            attachment_start_time = None
            filtered_position = (1.0 - BOTTLE_POSITION_LPF_ALPHA) * filtered_position + BOTTLE_POSITION_LPF_ALPHA * frame["bottle_position"]
            filtered_normal = unit((1.0 - NORMAL_LPF_ALPHA) * filtered_normal + NORMAL_LPF_ALPHA * frame["normal"])
        frame["bottle_position"] = filtered_position.copy()
        frame["normal"] = filtered_normal.copy()
        # The non-belief right hand continuously supports the bottle below its
        # midpoint from the side: it approaches perpendicular to the bottle
        # surface instead of along the bottle's longitudinal normal.
        if frame["right_rotation"] is not None:
            surface_outward, _ = orthogonal_basis(filtered_normal)
            contact_point = filtered_position - filtered_normal * 0.040
            frame["right_gripper_position"] = contact_point + surface_outward * 0.060
            frame["right_gripper_rotation"] = roll_gripper_90(rotation_with_approach_axis(-surface_outward))

    indices = np.linspace(0, len(rows) - 1, FRAME_COUNT, dtype=int)
    actual = np.vstack([item["actual_position"] for item in rows])
    reference = np.vstack([item["reference_position"] for item in rows])
    bottle = np.vstack([item["bottle_position"] for item in rows])
    right_points = np.vstack([item["right_gripper_position"] for item in rows if item.get("right_gripper_position") is not None])
    all_points = np.vstack((actual, reference, bottle, right_points))
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) * 0.78, 0.23)

    frames: list[Image.Image] = []
    for number, index in enumerate(indices, start=1):
        frame = rows[index]
        figure = plt.figure(figsize=(7.5, 6.5), dpi=120, facecolor="#f8fafc")
        axis = figure.add_axes((0.02, 0.02, 0.96, 0.86), projection="3d", facecolor="#f8fafc")
        axis.set_axis_off()
        axis.view_init(elev=19, azim=-57)
        axis.set_proj_type("ortho")
        axis._dist = 5.8
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius * 0.78, center[2] + radius * 0.78)
        axis.set_box_aspect((1, 1, 0.82))

        draw_bottle(axis, frame["bottle_position"], frame["normal"])
        arrow(axis, frame["bottle_position"] + frame["normal"] * 0.095, frame["normal"], "#f97316", 3.3, 0.105)

        trail_start = max(0, index - 24)
        trail = slice(trail_start, index + 1)
        axis.plot(*actual[trail].T, color="#2563eb", linewidth=2.5, alpha=0.45)
        axis.plot(*reference[trail].T, color="#22c55e", linewidth=2.5, alpha=0.45)

        axis.plot(*np.vstack((frame["actual_position"], frame["reference_position"])).T, color="#ef4444", linewidth=2.0, linestyle="--", alpha=0.9)
        axis.scatter(*frame["actual_position"], s=70, color="#2563eb", edgecolors="#ffffff", linewidths=1.2, depthshade=False)
        axis.scatter(*frame["reference_position"], s=82, color="#22c55e", edgecolors="#ffffff", linewidths=1.2, depthshade=False)
        draw_dex1_gripper(axis, frame["actual_position"], frame["actual_rotation"], frame["grasp_state"], frame["grasp_engaged"])
        if frame.get("right_gripper_position") is not None:
            axis.scatter(*frame["right_gripper_position"], s=42, color="#64748b", edgecolors="#ffffff", linewidths=1.0, depthshade=False)
            draw_dex1_gripper(axis, frame["right_gripper_position"], frame["right_gripper_rotation"], "holding", True)
        arrow(axis, frame["actual_position"], frame["actual_axis"], "#2563eb", 3.2)
        arrow(axis, frame["reference_position"], frame["reference_axis"], "#22c55e", 3.2)

        orientation_error = relation_error_degrees(frame["actual_axis"], frame["normal"], frame["relation"])
        figure.text(
            0.5,
            0.94,
            f"Normal relation error: {orientation_error:.1f}°",
            ha="center",
            va="center",
            color="#0f172a",
            fontsize=13,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.38", "facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.92},
        )
        relation_color = "#7c3aed" if frame["relation"] == "parallel" else "#c2410c"
        figure.text(
            0.065,
            0.105,
            frame["relation"].upper(),
            ha="left",
            va="center",
            color="#ffffff",
            fontsize=19,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": relation_color, "edgecolor": "none", "alpha": 0.96},
        )

        figure.canvas.draw()
        frames.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3]))
        plt.close(figure)

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=FRAME_DURATION_MS, loop=0, optimize=False, disposal=2)
    print(f"Wrote {output} ({FRAME_COUNT} frames, {FRAME_COUNT * FRAME_DURATION_MS / 1000:.1f} s)")
    print(f"Source span: {rows[0]['time']:.2f} to {rows[-1]['time']:.2f} s ({len(rows)} bottle-belief samples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE, help="PATH controller CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Output GIF path")
    arguments = parser.parse_args()
    main(arguments.source, arguments.output)
