import argparse
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

# Ensure clean imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.perception import PerceptionAgent
from agents.tracker import KalmanTracker
from agents.planner import PlannerAgent
from agents.controller import ControllerAgent
from eval.dataset_loader import SequenceDataset, FrameItem


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    perception_latency_ms: float,
    active_track_count: int,
    planner_output: Dict[str, Any],
    setpoints: Dict[str, Any],
    primary_target_id: Optional[int],
) -> np.ndarray:
    """
    Renders an on-screen visual HUD dashboard over the frame.
    Displays:
      - Current Frame, FPS, Perception Latency (ms)
      - Active Track Count, Primary Target ID
      - Planner Action State & Reasoning
      - 4-DOF Velocity Setpoints [vx, vy, vz, yaw_rate]
    """
    h, w = frame.shape[:2]

    # Dashboard overlay dimensions
    panel_w = min(680, w - 20)
    panel_h = 135
    x1, y1 = 10, 10
    x2, y2 = x1 + panel_w, y1 + panel_h

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 24, 28), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 75, 85), 1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Telemetry text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    line_spacing = 26
    font_thick = 1

    action = str(planner_output.get("action", "HOLD")).upper().strip()
    reasoning = planner_output.get("reasoning", "")
    if len(reasoning) > 42:
        reasoning = reasoning[:39] + "..."

    action_colors = {
        "SEARCH": (0, 215, 255),  # Amber/Yellow
        "CENTER": (255, 178, 50),  # Cyan/Sky
        "TRACK": (50, 255, 50),   # Green
        "HOLD": (180, 180, 180),  # Light Gray
    }
    action_color = action_colors.get(action, (255, 255, 255))

    sp = setpoints.get("setpoints", {})
    vx = sp.get("vx_m_s", 0.0)
    vy = sp.get("vy_m_s", 0.0)
    vz = sp.get("vz_m_s", 0.0)
    yaw_rate = sp.get("yaw_rate_rad_s", 0.0)

    # Line 1: Header / Performance Telemetry
    t1_left = f"FRAME: {frame_idx:04d} | FPS: {fps:.1f}"
    t1_right = f"PERCEPTION: {perception_latency_ms:.1f} ms"
    cv2.putText(frame, t1_left, (x1 + 12, y1 + 24), font, font_scale, (230, 230, 230), font_thick, cv2.LINE_AA)
    cv2.putText(frame, t1_right, (x1 + 320, y1 + 24), font, font_scale, (0, 255, 255), font_thick, cv2.LINE_AA)

    # Line 2: Tracking State
    target_str = f"#{primary_target_id}" if primary_target_id is not None else "NONE"
    t2 = f"TRACKS ACTIVE: {active_track_count:2d}  |  PRIMARY LOCK: {target_str}"
    cv2.putText(frame, t2, (x1 + 12, y1 + 24 + line_spacing), font, font_scale, (200, 220, 240), font_thick, cv2.LINE_AA)

    # Line 3: Planner State
    t3_prefix = "PLANNER: "
    cv2.putText(frame, t3_prefix, (x1 + 12, y1 + 24 + 2 * line_spacing), font, font_scale, (200, 200, 200), font_thick, cv2.LINE_AA)
    cv2.putText(frame, f"[{action}]", (x1 + 95, y1 + 24 + 2 * line_spacing), font, font_scale, action_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"- {reasoning}", (x1 + 195, y1 + 24 + 2 * line_spacing), font, font_scale - 0.05, (180, 190, 200), font_thick, cv2.LINE_AA)

    # Line 4: Control Setpoints
    t4 = f"SETPOINTS: Vx: {vx:+.2f} m/s | Vy: {vy:+.2f} m/s | Vz: {vz:+.2f} m/s | Yaw: {yaw_rate:+.2f} rad/s"
    cv2.putText(frame, t4, (x1 + 12, y1 + 24 + 3 * line_spacing), font, font_scale - 0.04, (100, 255, 100), font_thick, cv2.LINE_AA)

    return frame


def render_tracks(
    frame: np.ndarray,
    confirmed_tracks: List[Dict[str, Any]],
    primary_target_id: Optional[int],
) -> np.ndarray:
    """
    Overlays green bounding boxes with Track ID and velocity vectors for confirmed tracks,
    and a distinct red targeting reticle for the active primary target.
    """
    for track in confirmed_tracks:
        tid = track.get("track_id")
        label = track.get("label", "obj")
        bbox = track.get("bbox_pixel", [0, 0, 0, 0])
        vel = track.get("velocity_pixel_s", [0.0, 0.0])

        x1, y1, x2, y2 = [int(round(c)) for c in bbox]
        xc, yc = int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))
        vx, vy = vel[0], vel[1]

        is_primary = (tid is not None and primary_target_id is not None and tid == primary_target_id)

        if is_primary:
            # Red bounding box with thick highlight
            box_color = (0, 0, 255)
            box_thick = 3
            tag_bg = (0, 0, 220)
            tag_text = f"TARGET #{tid} [{label}]"
        else:
            # Green bounding box
            box_color = (0, 255, 0)
            box_thick = 2
            tag_bg = (0, 180, 0)
            tag_text = f"ID:{tid} {label}"

        # Draw main box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thick)

        # Label banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), baseline = cv2.getTextSize(tag_text, font, font_scale, 1)
        tag_y1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y1 + th + 6), tag_bg, -1)
        cv2.putText(frame, tag_text, (x1 + 3, tag_y1 + th + 2), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Velocity Vector Overlay (arrow from center)
        arrow_scale = 1.0  # Scale pixel velocity vector for visualization
        arrow_end_x = int(round(xc + vx * arrow_scale))
        arrow_end_y = int(round(yc + vy * arrow_scale))

        # Clamp arrow within frame bounds
        h, w = frame.shape[:2]
        arrow_end_x = max(0, min(w - 1, arrow_end_x))
        arrow_end_y = max(0, min(h - 1, arrow_end_y))

        vel_magnitude = (vx**2 + vy**2) ** 0.5
        if vel_magnitude > 2.0:
            arrow_color = (0, 140, 255) if not is_primary else (0, 255, 255)
            cv2.arrowedLine(frame, (xc, yc), (arrow_end_x, arrow_end_y), arrow_color, 2, tipLength=0.35)
            v_text = f"{vel_magnitude:.1f}px/s"
            cv2.putText(frame, v_text, (xc + 5, yc - 5), font, 0.38, arrow_color, 1, cv2.LINE_AA)

        # Draw target crosshair for primary target
        if is_primary:
            ch_len = 8
            cv2.line(frame, (xc - ch_len, yc), (xc + ch_len, yc), (0, 0, 255), 2)
            cv2.line(frame, (xc, yc - ch_len), (xc, yc + ch_len), (0, 0, 255), 2)

    return frame


def run_pipeline(
    input_path: str,
    output_path: str = "output_evaluation.mp4",
    planner_hz: float = 2.5,
    max_frames: Optional[int] = None,
    display: bool = False,
    device: Optional[str] = None,
) -> None:
    """
    Main orchestration loop coordinating PerceptionAgent, KalmanTracker, PlannerAgent, and ControllerAgent.
    """
    print("=" * 70)
    print("   UAV FLORENCE-2 MULTI-AGENT AUTONOMOUS FLIGHT PIPELINE   ")
    print("=" * 70)
    print(f"Input Sequence : {input_path}")
    print(f"Output Video   : {output_path}")
    print(f"Planner Rate   : {planner_hz:.1f} Hz")

    # 1. Initialize Dataset / Sequence loader
    dataset = SequenceDataset(sequence_path=input_path)
    total_frames = len(dataset)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    print(f"Total Frames to Process: {total_frames}")

    if total_frames == 0:
        print("[ERROR] No frames found in input sequence.")
        return

    # 2. Initialize the 4 Pipeline Agents
    print("\n[INIT] Initializing 4-Layer Multi-Agent Architecture...")
    perception_agent = PerceptionAgent(device=device)
    tracker = KalmanTracker(iou_threshold=0.3, n_min=2, m_max=5)
    
    planner_interval = 1.0 / max(0.1, planner_hz)
    planner_agent = PlannerAgent(min_call_interval_s=planner_interval)
    controller_agent = ControllerAgent()
    print("[INIT] All Agents Loaded Successfully.\n")

    # 3. Setup Video Writer
    first_frame_item = dataset.get_frame(0)
    sample_bgr = first_frame_item.get_bgr()
    if sample_bgr is None:
        raise RuntimeError("Could not decode sample frame for video writer initialization.")
    frame_h, frame_w = sample_bgr.shape[:2]

    fps_target = 15.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps_target, (frame_w, frame_h))

    # Runtime state
    active_plan = {
        "action": "SEARCH",
        "primary_target_id": None,
        "reasoning": "Pipeline booting, scanning airspace...",
    }
    last_setpoints = controller_agent.compute_setpoints(active_plan, primary_target=None)
    last_active_track_count = 0

    fps_rolling = fps_target
    dt_default = 1.0 / fps_target

    try:
        for idx in range(total_frames):
            frame_start_time = time.time()
            frame_item = dataset.get_frame(idx)
            frame_bgr = frame_item.get_bgr()

            if frame_bgr is None:
                continue

            # Convert BGR to PIL for PerceptionAgent
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)

            # Step A: Perception Agent (Florence-2 Object Detection)
            perception_result = perception_agent.process_frame(pil_image, task_prompt="<OD>")
            p_latency_ms = perception_result.get("latency_ms", 0.0)

            # Step B: Kalman Filter Tracking
            confirmed_tracks = tracker.update(perception_result, dt=dt_default)
            curr_track_count = len(confirmed_tracks)

            # Step C: Planner Agent (Throttled or triggered on state transition)
            state_changed = (curr_track_count > 0 and last_active_track_count == 0) or \
                            (curr_track_count == 0 and last_active_track_count > 0)

            active_plan = planner_agent.plan(confirmed_tracks, force_update=state_changed)
            last_active_track_count = curr_track_count

            # Determine primary target track object
            primary_id = active_plan.get("primary_target_id")
            primary_target = None
            if primary_id is not None:
                for trk in confirmed_tracks:
                    if trk.get("track_id") == primary_id:
                        primary_target = trk
                        break

            # Fallback target selection if ID not matched
            if primary_target is None and confirmed_tracks:
                primary_target = min(
                    confirmed_tracks,
                    key=lambda t: (
                        t.get("center_offset_normalized", [0.0, 0.0])[0] ** 2
                        + t.get("center_offset_normalized", [0.0, 0.0])[1] ** 2
                    )
                )
                primary_id = primary_target.get("track_id")

            # Step D: Controller Agent (Proportional 4-DOF Translation)
            last_setpoints = controller_agent.compute_setpoints(active_plan, primary_target=primary_target)

            # Step E: HUD and Track Rendering
            annotated_frame = frame_bgr.copy()
            annotated_frame = render_tracks(annotated_frame, confirmed_tracks, primary_target_id=primary_id)
            annotated_frame = draw_hud(
                annotated_frame,
                frame_idx=idx + 1,
                fps=fps_rolling,
                perception_latency_ms=p_latency_ms,
                active_track_count=curr_track_count,
                planner_output=active_plan,
                setpoints=last_setpoints,
                primary_target_id=primary_id,
            )

            # Write to output video
            video_writer.write(annotated_frame)

            # Optional Live Window
            if display:
                cv2.imshow("UAV Multi-Agent Live HUD", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] User requested termination via 'q'.")
                    break

            # Compute rolling FPS
            frame_elapsed = max(0.001, time.time() - frame_start_time)
            fps_instant = 1.0 / frame_elapsed
            fps_rolling = 0.85 * fps_rolling + 0.15 * fps_instant

            if (idx + 1) % 5 == 0 or (idx + 1) == total_frames:
                sp = last_setpoints.get("setpoints", {})
                print(
                    f"[{idx+1:04d}/{total_frames:04d}] "
                    f"Tracks: {curr_track_count:2d} | "
                    f"Plan: {active_plan.get('action'):<6} | "
                    f"Setpoints: [Vx:{sp.get('vx_m_s', 0):+.2f}, Vz:{sp.get('vz_m_s', 0):+.2f}, Yaw:{sp.get('yaw_rate_rad_s', 0):+.2f}] | "
                    f"Lat: {p_latency_ms:.1f}ms | FPS: {fps_rolling:.1f}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user.")
    finally:
        video_writer.release()
        dataset.close()
        if display:
            cv2.destroyAllWindows()

    print(f"\n[SUCCESS] Pipeline execution complete. Annotated video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UAV Multi-Agent Flight & Tracking Pipeline Orchestrator")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input sequence directory or video file.")
    parser.add_argument("--output", "-o", type=str, default="output_evaluation.mp4", help="Path to save output video.")
    parser.add_argument("--planner-hz", type=float, default=2.5, help="Invocation rate for CPU Planner Agent (Hz).")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Maximum number of frames to process.")
    parser.add_argument("--display", action="store_true", help="Display live OpenCV window during processing.")
    parser.add_argument("--device", type=str, default=None, help="Force device ('cuda' or 'cpu').")

    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        planner_hz=args.planner_hz,
        max_frames=args.max_frames,
        display=args.display,
        device=args.device,
    )


if __name__ == "__main__":
    main()

