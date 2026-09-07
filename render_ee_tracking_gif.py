"""Render an upper-body G1 end-effector tracking GIF from PATH telemetry.

The kinematic skeleton is parsed directly from the supplied locked-waist G1 MJCF XML.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


DATA_DIR = Path("public/data")
CSV_PATH = DATA_DIR / "path_static_ee_tracking.csv"
MODEL_PATH = DATA_DIR / "g1_29dof_lock_waist.xml"
OUTPUT_PATH = DATA_DIR / "20260805_174753_static_ee_static_upper.gif"

FRAME_COUNT = 150
FRAME_DURATION_MS = 80

# The telemetry follows the 29-DoF G1 convention: legs (0-11), waist (12-14),
# left arm (15-21), and right arm (22-28). The model locks waist roll/pitch.
JOINT_TO_Q_INDEX = {
    "waist_yaw_joint": 12,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}

UPPER_BODY_LINKS = {
    "pelvis",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
}


def vector(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.fromstring(value, sep=" ") if value else np.asarray(default, dtype=float)


def rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Convert MuJoCo's wxyz quaternion convention to a rotation matrix."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def upper_body_kinematics(
    root: ET.Element, joint_positions: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, str | None], dict[str, np.ndarray]]:
    positions: dict[str, np.ndarray] = {}
    parents: dict[str, str | None] = {}
    rotations: dict[str, np.ndarray] = {}

    def walk(body: ET.Element, parent_position: np.ndarray, parent_rotation: np.ndarray, parent_name: str | None) -> None:
        name = body.get("name")
        local_position = vector(body.get("pos"), (0.0, 0.0, 0.0))
        local_rotation = rotation_from_quaternion(vector(body.get("quat"), (1.0, 0.0, 0.0, 0.0)))
        position = parent_position + parent_rotation @ local_position
        rotation = parent_rotation @ local_rotation

        for joint in body.findall("joint"):
            index = JOINT_TO_Q_INDEX.get(joint.get("name", ""))
            if index is not None:
                rotation = rotation @ rotation_about_axis(vector(joint.get("axis"), (0.0, 0.0, 1.0)), joint_positions[index])

        if name in UPPER_BODY_LINKS:
            positions[name] = position
            parents[name] = parent_name if parent_name in UPPER_BODY_LINKS else None
            rotations[name] = rotation

        for child in body.findall("body"):
            walk(child, position, rotation, name)

    pelvis = root.find("./worldbody/body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError("Could not find the pelvis body in the MJCF model.")
    # The CSV uses the pelvis frame, so place the pelvis at the visual origin.
    pelvis_copy = ET.fromstring(ET.tostring(pelvis, encoding="unicode"))
    pelvis_copy.set("pos", "0 0 0")
    walk(pelvis_copy, np.zeros(3), np.eye(3), None)
    return positions, parents, rotations


def draw_segment(axis, start: np.ndarray, end: np.ndarray, color: str, width: float, alpha: float = 1.0) -> None:
    axis.plot(*np.column_stack((start, end)), color=color, linewidth=width, alpha=alpha, solid_capstyle="round")


def main() -> None:
    model_root = ET.parse(MODEL_PATH).getroot()
    telemetry = pd.read_csv(CSV_PATH)
    q_columns = [f"body_q_measured_{index:02d}" for index in range(29)]
    sample_indices = np.linspace(0, len(telemetry) - 1, FRAME_COUNT, dtype=int)

    measured = telemetry[["left_wrist_position_00", "left_wrist_position_01", "left_wrist_position_02"]].to_numpy(float)
    target = telemetry[["terminal_target_position_00", "terminal_target_position_01", "terminal_target_position_02"]].to_numpy(float)
    error = telemetry["active_target_error_m"].to_numpy(float)

    poses = []
    parents = None
    for index in sample_indices:
        pose, parents, rotations = upper_body_kinematics(model_root, telemetry.loc[index, q_columns].to_numpy(float))
        poses.append((pose, rotations))
    assert parents is not None

    all_points = np.vstack(
        [
            measured[sample_indices],
            target[sample_indices],
            *[np.vstack(list(pose.values())) for pose, _ in poses],
        ]
    )
    center = all_points.mean(axis=0)
    radius = max(np.ptp(all_points, axis=0).max() * 0.47, 0.22)

    frames: list[Image.Image] = []
    for frame_number, (sample_index, (pose, rotations)) in enumerate(zip(sample_indices, poses), start=1):
        figure = plt.figure(figsize=(7.4, 6.4), dpi=120, facecolor="#f8fafc")
        axis = figure.add_axes((0.02, 0.02, 0.96, 0.86), projection="3d", facecolor="#f8fafc")
        axis.set_axis_off()
        axis.view_init(elev=18, azim=-62)
        axis.set_proj_type("ortho")
        axis._dist = 7.4  # Tighter orthographic framing: roughly 1.35x larger than Matplotlib's default view.
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius * 0.8, center[2] + radius * 0.8)
        axis.set_box_aspect((1, 1, 0.85))

        # A restrained ground plane makes depth and horizontal motion immediately legible.
        grid = np.linspace(-radius, radius, 2)
        ground_x, ground_y = np.meshgrid(center[0] + grid, center[1] + grid)
        ground_z = np.full_like(ground_x, center[2] - radius * 0.56)
        axis.plot_surface(ground_x, ground_y, ground_z, color="#e2e8f0", alpha=0.24, shade=False)

        for name, point in pose.items():
            parent = parents[name]
            if parent is None:
                continue
            color = "#2563eb" if name.startswith("left_") else "#475569"
            width = 5.0 if name.startswith("left_") else 3.8
            draw_segment(axis, pose[parent], point, color, width)

        torso = pose["torso_link"]
        axis.scatter(*torso, s=100, color="#0f172a", depthshade=False)
        head = torso + rotations["torso_link"] @ np.array([0.0039635, 0.0, -0.054])
        axis.scatter(*head, s=72, color="#0f172a", depthshade=False)

        trail_start = max(0, frame_number - 26)
        trail_indices = sample_indices[trail_start:frame_number]
        if len(trail_indices) > 1:
            axis.plot(*measured[trail_indices].T, color="#22c55e", linewidth=2.6, alpha=0.58)

        actual = measured[sample_index]
        goal = target[sample_index]
        draw_segment(axis, pose["left_wrist_yaw_link"], actual, "#2563eb", 5.0)
        draw_segment(axis, actual, goal, "#ef4444", 3.2)
        axis.scatter(*actual, s=88, color="#16a34a", edgecolors="#ffffff", linewidths=1.4, depthshade=False)
        axis.scatter(*goal, s=150, color="#f97316", marker="*", edgecolors="#ffffff", linewidths=0.8, depthshade=False)

        figure.text(
            0.5,
            0.94,
            f"EE error: {error[sample_index] * 100:.1f} cm",
            ha="center",
            va="center",
            color="#b91c1c",
            fontsize=21,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.38", "facecolor": "#fff7ed", "edgecolor": "none", "alpha": 0.92},
        )

        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3]))
        plt.close(figure)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=False,
        disposal=2,
    )

    model_wrist = np.vstack(
        [pose["left_wrist_yaw_link"] + rotations["left_wrist_yaw_link"] @ np.array([0.0415, 0.003, 0.0]) for pose, rotations in poses]
    )
    alignment_error = np.linalg.norm(model_wrist - measured[sample_indices], axis=1)
    print(f"Wrote {OUTPUT_PATH} ({FRAME_COUNT} frames, {FRAME_COUNT * FRAME_DURATION_MS / 1000:.1f} s)")
    print(f"Mean model-to-logged left-wrist offset: {alignment_error.mean() * 100:.2f} cm")


if __name__ == "__main__":
    main()
