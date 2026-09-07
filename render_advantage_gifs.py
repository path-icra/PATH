"""Create compact visual explanations of PATH contact and orientation advantages.

The figures use the clean bottle-manipulation segment recorded in trial_013.
"""

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
from matplotlib.patches import FancyBboxPatch


SOURCE = Path(r"C:\studioprojects\PATH-G1\record\controller\PATH\trial_013.csv")
OUTPUT_DIR = Path("public/data")
CONTACT_OUTPUT = OUTPUT_DIR / "path_contact_advantage_trial_013.gif"
ORIENTATION_OUTPUT = OUTPUT_DIR / "path_orientation_advantage_trial_013.gif"
HAND_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])
FRAME_COUNT = 108
FRAME_DURATION_MS = 78

BLUE, GREEN, ORANGE, PURPLE, SLATE, MUTED = "#2563eb", "#16a34a", "#f97316", "#7c3aed", "#334155", "#94a3b8"


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-9)


def pose(value: str) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(value)
    return np.asarray(payload["position_m"], dtype=float), np.asarray(payload["orientation"], dtype=float).reshape(3, 3)


def valid_rows(source: Path = SOURCE, start_time: float | None = None, end_time: float | None = None) -> list[dict]:
    rows: list[dict] = []
    for _, row in pd.read_csv(source).iterrows():
        elapsed = float(row["elapsed_s"])
        if (start_time is not None and elapsed < start_time) or (end_time is not None and elapsed > end_time):
            continue
        belief_raw, objects_raw, hand_raw = row["left_belief_target_json"], row["left_objects_json"], row["left_ee_json"]
        if not all(isinstance(value, str) and value not in {"", "null", "[]"} for value in (belief_raw, objects_raw, hand_raw)):
            continue
        belief, objects = json.loads(belief_raw), json.loads(objects_raw)
        if "bottle" not in str(belief.get("object_id", "")).lower() or not objects:
            continue
        belief_normal = np.asarray(belief.get("orientation_axis"), dtype=float)
        if belief_normal.shape != (3,) or not np.isfinite(belief_normal).all():
            continue
        bottle = next((item for item in objects if "bottle" in str(item.get("label", item.get("object_id", ""))).lower()), None)
        if not bottle or not bottle.get("contacts"):
            continue
        hand_position, hand_rotation = pose(hand_raw)
        contacts = bottle["contacts"]
        rows.append(
            {
                "time": elapsed,
                "hand": hand_position,
                "axis": unit(hand_rotation @ HAND_AXIS_LOCAL),
                "normal": unit(belief_normal),
                "relation": str(belief.get("relation", "perpendicular")),
                "object_posterior": float(bottle.get("posterior", 0.0)),
                "contacts": contacts,
            }
        )
    if not rows:
        raise RuntimeError("No bottle-contact belief frames found in trial_013.")
    # Plane normals are unsigned. Retain one visually stable sign over the clip.
    for index in range(1, len(rows)):
        if np.dot(rows[index - 1]["normal"], rows[index]["normal"]) < 0:
            rows[index]["normal"] *= -1
    assign_contact_tracks(rows)
    return rows


def assign_contact_tracks(rows: list[dict]) -> None:
    """Associate nearby contact samples over time, then smooth their evidence."""
    tracks: list[dict] = []
    smoothed_evidence: dict[int, float] = {}
    for frame in rows:
        claimed: set[int] = set()
        for contact in frame["contacts"]:
            position = np.asarray(contact["position_m"], dtype=float)
            distances = [np.linalg.norm(position - track["center"]) if track["id"] not in claimed else np.inf for track in tracks]
            nearest = int(np.argmin(distances)) if distances else -1
            if nearest < 0 or distances[nearest] > 0.014:
                track_id = len(tracks)
                tracks.append({"id": track_id, "center": position.copy()})
            else:
                track_id = tracks[nearest]["id"]
                tracks[nearest]["center"] = 0.90 * tracks[nearest]["center"] + 0.10 * position
            claimed.add(track_id)
            raw_evidence = float(contact.get("posterior", 0.0))
            smoothed_evidence[track_id] = 0.82 * smoothed_evidence.get(track_id, raw_evidence) + 0.18 * raw_evidence
            contact["path_track_id"] = track_id
            contact["path_smoothed_evidence"] = smoothed_evidence[track_id]


def capture(figure: plt.Figure) -> Image.Image:
    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3])
    plt.close(figure)
    return image


def rounded_box(axis, x: float, y: float, width: float, height: float, *, facecolor: str, edgecolor: str = "none", linewidth: float = 0.0, radius: float = 0.030, zorder: float = -2) -> FancyBboxPatch:
    """Add a softly rounded UI container in axes-relative coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=axis.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        mutation_aspect=1,
        clip_on=False,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def text_box(figure: plt.Figure, x: float, y: float, text: str, color: str, size: float = 12) -> None:
    figure.text(x, y, text, color="white", fontsize=size, fontweight="bold", ha="left", va="center",
                bbox={"boxstyle": "round,pad=0.42", "facecolor": color, "edgecolor": "none"})


def score_bar(axis, y: float, label: str, score: float | None, color: str, selected: bool = False) -> None:
    axis.text(0.065, y, label, transform=axis.transAxes, fontsize=12.0, color=SLATE, va="center", fontweight="bold" if selected else "normal")
    if score is None:
        axis.text(0.945, y, "gated", transform=axis.transAxes, fontsize=11.0, color=MUTED, va="center", ha="right", style="italic")
        axis.plot([0.50, 0.945], [y, y], transform=axis.transAxes, color="#e2e8f0", linewidth=10, solid_capstyle="round")
        return
    axis.plot([0.50, 0.945], [y, y], transform=axis.transAxes, color="#e2e8f0", linewidth=10, solid_capstyle="round")
    axis.plot([0.50, 0.50 + 0.445 * score], [y, y], transform=axis.transAxes, color=color, linewidth=10, solid_capstyle="round")


def contact_scores(frame: dict, velocity: np.ndarray) -> tuple[list[dict], np.ndarray]:
    candidates = []
    for contact in frame["contacts"]:
        position = np.asarray(contact["position_m"], dtype=float)
        displacement = position - frame["hand"]
        distance = float(np.linalg.norm(displacement))
        proximity = float(np.exp(-distance / 0.20))
        direction = float(np.clip((np.dot(unit(velocity), unit(displacement)) + 1.0) / 2.0, 0.0, 1.0))
        reach = float(distance < 0.55)
        posterior = float(contact.get("posterior", 0.0))
        candidates.append(
            {
                "position": position,
                "proximity": proximity,
                "direction": direction,
                "reach": reach,
                "posterior": posterior,
                "evidence": float(contact.get("path_smoothed_evidence", posterior)),
                "track_id": int(contact.get("path_track_id", 0)),
                "selected": bool(contact.get("selected", False)),
            }
        )
    # Posterior is the logged, temporally smoothed selection evidence. The individual
    # bars are instantaneous interpretable components; alignment is shown gated because
    # this particular recording contains no reliable local surface normals.
    return candidates, np.mean([item["position"] for item in candidates], axis=0)


def render_contact(rows: list[dict], output: Path = CONTACT_OUTPUT, frame_count: int = FRAME_COUNT) -> None:
    indices = np.linspace(4, len(rows) - 1, frame_count, dtype=int)
    all_points = np.vstack([item["hand"] for item in rows] + [np.asarray(contact["position_m"], dtype=float) for item in rows for contact in item["contacts"]])
    center = all_points.mean(axis=0)
    scale = max(float(np.ptp(all_points[:, :2], axis=0).max()) * 0.62, 0.18)
    frames: list[Image.Image] = []
    for frame_no, index in enumerate(indices):
        frame, before = rows[index], rows[max(0, index - 6)]
        velocity = frame["hand"] - before["hand"]
        if np.linalg.norm(velocity) < 1.0e-5:
            velocity = np.array([1.0, 0.0, 0.0])
        candidates, object_center = contact_scores(frame, velocity)
        selected = next((item for item in candidates if item["selected"]), max(candidates, key=lambda item: item["evidence"]))
        evidence_ranked = sorted(candidates, key=lambda item: item["evidence"], reverse=True)
        candidate_colors: dict[int, str] = {selected["track_id"]: GREEN}
        unselected = [item for item in evidence_ranked if item["track_id"] != selected["track_id"]]
        for candidate in unselected[:2]:
            candidate_colors[candidate["track_id"]] = ORANGE
        for candidate in unselected[2:]:
            candidate_colors[candidate["track_id"]] = "#c2410c"
        ranked = [selected] + sorted(unselected, key=lambda item: item["evidence"], reverse=True)[:2]

        figure = plt.figure(figsize=(8.8, 5.6), dpi=120, facecolor="#f8fafc")
        scene = figure.add_axes((0.055, 0.075, 0.405, 0.85), facecolor="none")
        panel = figure.add_axes((0.515, 0.075, 0.43, 0.85), facecolor="none")
        for axis in (scene, panel):
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values(): spine.set_visible(False)
        rounded_box(scene, 0.0, 0.0, 1.0, 1.0, facecolor="#ffffff", radius=0.045)

        scene.set_xlim(center[0] - scale, center[0] + scale)
        scene.set_ylim(center[1] - scale, center[1] + scale)
        scene.set_aspect("equal")
        # Bottle footprint and contact hypotheses.
        scene.add_patch(plt.Circle(object_center[:2], 0.051, color="#e2e8f0", ec=SLATE, lw=1.8, zorder=0))
        for candidate in candidates:
            color = candidate_colors[candidate["track_id"]]
            radius = 0.012 if candidate["selected"] else 0.008
            scene.add_patch(plt.Circle(candidate["position"][:2], radius, color=color, ec="white", lw=1.0, zorder=3))
        history = np.vstack([item["hand"] for item in rows[max(0, index - 32):index + 1]])
        scene.plot(history[:, 0], history[:, 1], color=BLUE, linewidth=2.6, alpha=0.40)
        scene.scatter(*frame["hand"][:2], s=130, color=BLUE, edgecolors="white", linewidths=1.5, zorder=4)
        scene.arrow(frame["hand"][0], frame["hand"][1], velocity[0] * 0.55, velocity[1] * 0.55, color=BLUE, width=0.0036, head_width=0.020, length_includes_head=True, zorder=4)
        scene.plot([frame["hand"][0], selected["position"][0]], [frame["hand"][1], selected["position"][1]], color=GREEN, linewidth=2.0, linestyle="--", alpha=0.9)
        
        for candidate_no, candidate in enumerate(ranked):
            base = 0.805 - candidate_no * 0.285
            winner = candidate["selected"]
            candidate_color = candidate_colors[candidate["track_id"]]
            card_fill = "#ecfdf5" if winner else "#fff7ed"
            card_y, card_height = base - 0.150, 0.230
            rounded_box(panel, 0.010, card_y, 0.980, card_height, facecolor=card_fill, radius=0.030, zorder=-1)
            if winner:
                rounded_box(panel, 0.010, card_y, 0.980, card_height, facecolor="none", edgecolor=GREEN, linewidth=3.5, radius=0.030, zorder=2)
            panel.plot([0.035, 0.035], [card_y + 0.030, card_y + card_height - 0.030], transform=panel.transAxes, color=candidate_color, linewidth=6.5, solid_capstyle="round", zorder=1)
            panel.text(0.060, base + 0.040, f"A(c)  {candidate['evidence']:.2f}", transform=panel.transAxes, fontsize=12.4, color=candidate_color, fontweight="bold")
            score_bar(panel, base - 0.012, "Near", candidate["proximity"], GREEN, winner)
            score_bar(panel, base - 0.054, "Moving toward", candidate["direction"], BLUE, winner)
            score_bar(panel, base - 0.096, "Reachable", candidate["reach"], PURPLE, winner)
            score_bar(panel, base - 0.138, "Surface-aligned", None, ORANGE, winner)
        frames.append(capture(figure))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=FRAME_DURATION_MS, loop=0, optimize=False, disposal=2)
    print(f"Wrote {output} ({len(frames)} frames)")


def draw_relation_card(axis, origin: tuple[float, float], title: str, error: float, hold: float, convergence: float, selected: bool) -> None:
    x, y = origin
    edge = GREEN if selected else "#cbd5e1"
    rounded_box(axis, x, y, 0.470, 0.84, facecolor="#ffffff", edgecolor=edge, linewidth=3.1 if selected else 1.5, radius=0.032, zorder=-1)
    axis.text(x + 0.045, y + 0.735, title, transform=axis.transAxes, fontsize=15.0 if title == "PERPENDICULAR" else 16.5, color=GREEN if selected else SLATE, fontweight="bold")
    axis.text(x + 0.045, y + 0.645, "SELECTED" if selected else "hypothesis", transform=axis.transAxes, fontsize=11.3, color=GREEN if selected else MUTED, fontweight="bold")
    advantage = float(np.clip(0.52 * (1 - error) + 0.28 * hold + 0.20 * convergence, 0.0, 1.0))
    axis.text(x + 0.045, y + 0.540, "ADVANTAGE", transform=axis.transAxes, fontsize=10.8, color=SLATE, fontweight="bold")
    axis.plot([x + 0.045, x + 0.425], [y + 0.475, y + 0.475], transform=axis.transAxes, color="#e2e8f0", linewidth=13, solid_capstyle="round")
    axis.plot([x + 0.045, x + 0.045 + 0.380 * advantage], [y + 0.475, y + 0.475], transform=axis.transAxes, color=GREEN if selected else BLUE, linewidth=13, solid_capstyle="round")
    for label_offset, bar_offset, label, value, color in ((0.360, 0.305, "Alignment", 1 - error, GREEN), (0.225, 0.170, "Stable hold", hold, PURPLE), (0.090, 0.035, "Converging", convergence, BLUE)):
        axis.text(x + 0.045, y + label_offset, label, transform=axis.transAxes, fontsize=10.8, color=SLATE, va="center")
        axis.plot([x + 0.045, x + 0.425], [y + bar_offset, y + bar_offset], transform=axis.transAxes, color="#e2e8f0", linewidth=9, solid_capstyle="round")
        axis.plot([x + 0.045, x + 0.045 + 0.380 * value], [y + bar_offset, y + bar_offset], transform=axis.transAxes, color=color, linewidth=9, solid_capstyle="round")


def render_orientation(rows: list[dict], output: Path = ORIENTATION_OUTPUT, frame_count: int = FRAME_COUNT, hold_end_frames: int = 0) -> None:
    times = np.asarray([item["time"] for item in rows])
    hand_axis = np.vstack([item["axis"] for item in rows])
    normal = np.vstack([item["normal"] for item in rows])
    # Filter only the visual normal to mirror the reference-generation display.
    for index in range(1, len(normal)):
        normal[index] = unit(0.94 * normal[index - 1] + 0.06 * normal[index])
    mu = np.sum(hand_axis * normal, axis=1)
    errors = {"parallel": 1 - mu**2, "perpendicular": mu**2}
    axis_rate = np.linalg.norm(np.gradient(hand_axis, times, axis=0), axis=1)
    derivatives = {key: np.gradient(value, times) for key, value in errors.items()}
    indices = np.linspace(3, len(rows) - 3, frame_count, dtype=int)
    frames: list[Image.Image] = []
    for index in indices:
        relation = rows[index]["relation"]
        components = {}
        for key in ("parallel", "perpendicular"):
            error = float(np.clip(errors[key][index], 0, 1))
            hold = float((1 - error) if axis_rate[index] < 0.55 else 0.0)
            convergence = float(np.clip(-derivatives[key][index] / 1.2, 0, 1))
            components[key] = error, hold, convergence
        figure = plt.figure(figsize=(8.8, 5.6), dpi=120, facecolor="#f8fafc")
        diagram = figure.add_axes((0.055, 0.075, 0.320, 0.85), facecolor="none")
        cards = figure.add_axes((0.400, 0.075, 0.55, 0.85), facecolor="none")
        for axis in (diagram, cards):
            axis.set_xticks([]); axis.set_yticks([])
            for spine in axis.spines.values(): spine.set_visible(False)
        rounded_box(diagram, 0.0, 0.0, 1.0, 1.0, facecolor="#ffffff", radius=0.045)
        diagram.set_xlim(-1.18, 1.18); diagram.set_ylim(-1.18, 1.18); diagram.set_aspect("equal")
        # Draw the schematic from the true 3-D unsigned axis angle instead of
        # a camera-plane projection, which can visually contradict the belief.
        angle = float(np.degrees(np.arccos(np.clip(abs(mu[index]), 0, 1))))
        radians = np.radians(angle)
        n = np.array([0.0, 1.0])
        h = np.array([np.sin(radians), np.cos(radians)])
        diagram.add_patch(plt.Circle((0, 0), 0.18, color="#e2e8f0", ec=SLATE, lw=1.6))
        
        diagram.arrow(0, 0, h[0] * 0.80, h[1] * 0.80, color=BLUE, width=0.026, head_width=0.12, length_includes_head=True)
        diagram.arrow(0, 0, n[0] * 0.80, n[1] * 0.80, color=ORANGE, width=0.026, head_width=0.12, length_includes_head=True)
        diagram.text(0, -0.98, f"axis angle  {angle:.0f}°", ha="center", fontsize=14, color=SLATE, fontweight="bold")
        for card_index, key in enumerate(("parallel", "perpendicular")):
            draw_relation_card(cards, (0.015 + card_index * 0.500, 0.080), key.upper(), *components[key], key == relation)
        frames.append(capture(figure))
    if hold_end_frames:
        frames.extend(frames[-1].copy() for _ in range(hold_end_frames))
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=FRAME_DURATION_MS, loop=0, optimize=False, disposal=2)
    print(f"Wrote {output} ({len(frames)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("contact", "orientation"), help="Render one explanatory GIF only.")
    parser.add_argument("--source", type=Path, default=SOURCE, help="PATH controller CSV")
    parser.add_argument("--start", type=float, help="Optional source-clip start time in seconds")
    parser.add_argument("--end", type=float, help="Optional source-clip end time in seconds")
    parser.add_argument("--contact-output", type=Path, default=CONTACT_OUTPUT, help="Contact GIF output path")
    parser.add_argument("--orientation-output", type=Path, default=ORIENTATION_OUTPUT, help="Orientation GIF output path")
    parser.add_argument("--frames", type=int, default=FRAME_COUNT, help="Number of GIF frames to render")
    parser.add_argument("--hold-end-frames", type=int, default=0, help="Extra still frames appended after the final valid pose")
    arguments = parser.parse_args()
    data = valid_rows(arguments.source, arguments.start, arguments.end)
    if arguments.only in {None, "contact"}:
        render_contact(data, arguments.contact_output, arguments.frames)
    if arguments.only in {None, "orientation"}:
        render_orientation(data, arguments.orientation_output, arguments.frames, arguments.hold_end_frames)
