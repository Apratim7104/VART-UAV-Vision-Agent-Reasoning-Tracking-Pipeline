import argparse
import os
import sys
import time
from typing import Optional
import cv2
import numpy as np
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.perception import PerceptionAgent
from agents.tracker import KalmanTracker
from agents.planner import PlannerAgent
from agents.controller import ControllerAgent
from eval.dataset_loader import SequenceDataset
from eval.metrics import BenchmarkEvaluator
from run_pipeline import render_tracks, draw_hud


def print_evaluation_summary(report_dict: dict) -> None:
    """Prints a clean ASCII summary table of benchmark metrics."""
    summary = report_dict.get("summary", {})
    perception = report_dict.get("perception", {})
    tracking = report_dict.get("tracking", {})
    planner = report_dict.get("planner", {})

    print("\n" + "=" * 65)
    print("      UAV FLORENCE-2 MULTI-AGENT BENCHMARK EVALUATION REPORT     ")
    print("=" * 65)
    print(f" Total Frames Evaluated : {summary.get('total_frames', 0)}")
    print(f" Total Elapsed Time     : {summary.get('total_duration_s', 0.0):.2f} s")
    print(f" System Throughput      : {summary.get('system_average_fps', 0.0):.2f} FPS")
    print("-" * 65)
    print(" [PERCEPTION METRICS (Florence-2-base)]")
    print(f"  - Mean Inference Latency : {perception.get('mean_latency_ms', 0.0):.2f} ms")
    print(f"  - Median Latency         : {perception.get('median_latency_ms', 0.0):.2f} ms")
    print(f"  - Std Latency Deviation  : {perception.get('std_latency_ms', 0.0):.2f} ms")
    print(f"  - Min / Max Latency      : {perception.get('min_latency_ms', 0.0):.2f} ms / {perception.get('max_latency_ms', 0.0):.2f} ms")
    print(f"  - Total Detections       : {perception.get('total_detections', 0)}")
    print("-" * 65)
    print(" [TRACKING METRICS (Continuous 2D Kalman + Hungarian)]")
    print(f"  - Mean IoU (mIoU)        : {tracking.get('mean_iou', 0.0):.4f}")
    print(f"  - ID Switches (IDSW)     : {tracking.get('id_switches', 0)}")
    print(f"  - Track Fragmentations   : {tracking.get('fragmentations', 0)}")
    print(f"  - Mostly Tracked Ratio   : {tracking.get('mostly_tracked_ratio', 0.0) * 100:.1f}%")
    print(f"  - Mostly Lost Ratio      : {tracking.get('mostly_lost_ratio', 0.0) * 100:.1f}%")
    print(f"  - Ground Truth Targets   : {tracking.get('total_gt_targets', 0)}")
    print(f"  - Predicted Tracks Total : {tracking.get('unique_predicted_tracks', 0)}")
    print("-" * 65)
    print(" [PLANNER METRICS (Ollama / Qwen2.5-1.5B)]")
    print(f"  - Mean Decision Latency  : {planner.get('mean_latency_ms', 0.0):.2f} ms")
    print(f"  - Total Invocations      : {planner.get('total_calls', 0)}")
    dist = planner.get("action_distribution_pct", {})
    dist_str = ", ".join([f"{k}: {v:.1f}%" for k, v in dist.items()]) if dist else "N/A"
    print(f"  - Action Distribution    : {dist_str}")
    print("=" * 65 + "\n")


def evaluate_sequence(
    sequence_dir: str,
    output_dir: str = "eval_output",
    gt_path: Optional[str] = None,
    save_video: bool = True,
    planner_hz: float = 2.5,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
    display: bool = False,
    device: Optional[str] = None,
) -> str:
    """
    Executes end-to-end evaluation on a VisDrone or UAVDT sequence.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_json_path = os.path.join(output_dir, "eval_results.json")
    video_output_path = os.path.join(output_dir, "output_evaluation.mp4")

    # 1. Load Dataset Sequence
    dataset = SequenceDataset(sequence_path=sequence_dir, gt_path=gt_path)
    total_frames = len(dataset)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    if total_frames == 0:
        raise ValueError(f"Sequence {sequence_dir} has no valid frames.")

    print(f"\n[EVAL] Starting evaluation on: {sequence_dir}")
    print(f"[EVAL] Frames: {total_frames} | Save Video: {save_video} | Output Dir: {output_dir}")

    # 2. Initialize Agents & Evaluator
    perception_agent = PerceptionAgent(device=device)
    tracker = KalmanTracker(iou_threshold=0.3, n_min=2, m_max=5)
    planner_interval = 1.0 / max(0.1, planner_hz)
    planner_agent = PlannerAgent(min_call_interval_s=planner_interval)
    controller_agent = ControllerAgent()

    evaluator = BenchmarkEvaluator(match_iou_thresh=0.3)

    # 3. Initialize Optional Video Writer
    video_writer = None
    if save_video:
        sample_frame = dataset.get_frame(0).get_bgr()
        if sample_frame is not None:
            fh, fw = sample_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_output_path, fourcc, 15.0, (fw, fh))

    active_plan = {
        "action": "SEARCH",
        "primary_target_id": None,
        "reasoning": "Evaluator initialized.",
    }
    last_setpoints = controller_agent.compute_setpoints(active_plan, primary_target=None)
    last_active_track_count = 0
    fps_rolling = 15.0
    dt_step = 1.0 / 15.0

    try:
        for idx in range(start_frame, total_frames):
            t_frame_start = time.time()
            frame_item = dataset.get_frame(idx)
            frame_bgr = frame_item.get_bgr()

            if frame_bgr is None:
                continue

            # Convert BGR to PIL for PerceptionAgent
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Perception Step
            p_res = perception_agent.process_frame(pil_img, task_prompt="<OD>")
            p_lat = p_res.get("latency_ms", 0.0)
            num_dets = p_res.get("num_detections", 0)

            # Tracking Step
            confirmed_tracks = tracker.update(p_res, dt=dt_step)
            curr_track_count = len(confirmed_tracks)

            # Planner Step (throttled or on state transition)
            state_transition = (curr_track_count > 0 and last_active_track_count == 0) or \
                               (curr_track_count == 0 and last_active_track_count > 0)

            t_plan_start = time.time()
            active_plan = planner_agent.plan(confirmed_tracks, force_update=state_transition)
            plan_lat_ms = (time.time() - t_plan_start) * 1000.0
            last_active_track_count = curr_track_count

            # Primary Target Resolution
            primary_id = active_plan.get("primary_target_id")
            primary_target = None
            if primary_id is not None:
                for trk in confirmed_tracks:
                    if trk.get("track_id") == primary_id:
                        primary_target = trk
                        break

            if primary_target is None and confirmed_tracks:
                primary_target = min(
                    confirmed_tracks,
                    key=lambda t: (
                        t.get("center_offset_normalized", [0.0, 0.0])[0] ** 2
                        + t.get("center_offset_normalized", [0.0, 0.0])[1] ** 2
                    )
                )
                primary_id = primary_target.get("track_id")

            # Controller Step
            last_setpoints = controller_agent.compute_setpoints(active_plan, primary_target=primary_target)

            # Record Benchmark Metrics
            evaluator.step(
                frame_idx=idx + 1,
                perception_latency_ms=p_lat,
                num_detections=num_dets,
                predicted_tracks=confirmed_tracks,
                ground_truth_boxes=frame_item.gt_boxes,
                planner_output=active_plan,
                planner_latency_ms=plan_lat_ms if state_transition or (time.time() - planner_agent.last_call_time < 0.1) else None,
            )

            # Optional HUD Video Render
            if video_writer is not None:
                ann_frame = frame_bgr.copy()
                ann_frame = render_tracks(ann_frame, confirmed_tracks, primary_target_id=primary_id)
                ann_frame = draw_hud(
                    ann_frame,
                    frame_idx=idx + 1,
                    fps=fps_rolling,
                    perception_latency_ms=p_lat,
                    active_track_count=curr_track_count,
                    planner_output=active_plan,
                    setpoints=last_setpoints,
                    primary_target_id=primary_id,
                )
                video_writer.write(ann_frame)

                if display:
                    cv2.imshow("VisDrone Evaluation HUD", ann_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            # Calculate FPS
            elapsed = max(0.001, time.time() - t_frame_start)
            fps_rolling = 0.85 * fps_rolling + 0.15 * (1.0 / elapsed)

            if (idx + 1) % 10 == 0 or (idx + 1) == total_frames:
                gt_count = len(frame_item.gt_boxes)
                print(f"[EVAL Frame {idx+1:03d}/{total_frames:03d}] GT: {gt_count} | Pred Tracks: {curr_track_count} | Action: {active_plan.get('action')} | Lat: {p_lat:.1f}ms")

    finally:
        if video_writer is not None:
            video_writer.release()
        dataset.close()
        if display:
            cv2.destroyAllWindows()

    # Finalize and Save Results
    evaluator.save_json(results_json_path)
    report = evaluator.finalize()
    print_evaluation_summary(report.to_dict())

    print(f"[EVAL COMPLETE] Results successfully saved to: {results_json_path}")
    if save_video:
        print(f"[EVAL COMPLETE] Video successfully saved to: {video_output_path}")

    return results_json_path


def main():
    parser = argparse.ArgumentParser(description="VisDrone / UAVDT Multi-Agent Benchmark Evaluator")
    parser.add_argument("--sequence-dir", "-s", type=str, required=True, help="Path to VisDrone sequence folder.")
    parser.add_argument("--output-dir", "-o", type=str, default="eval_output", help="Directory for results & video.")
    parser.add_argument("--gt-path", "-g", type=str, default=None, help="Explicit path to gt.txt if not in sequence-dir.")
    parser.add_argument("--save-video", action="store_true", default=True, help="Save annotated evaluation video.")
    parser.add_argument("--no-video", dest="save_video", action="store_false", help="Disable video recording.")
    parser.add_argument("--planner-hz", type=float, default=2.5, help="Planner execution frequency in Hz.")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Max frames to process.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to resume evaluation from.")
    parser.add_argument("--display", action="store_true", help="Display live OpenCV window during evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Hardware compute device ('cuda' or 'cpu').")

    args = parser.parse_args()
    evaluate_sequence(
        sequence_dir=args.sequence_dir,
        output_dir=args.output_dir,
        gt_path=args.gt_path,
        save_video=args.save_video,
        planner_hz=args.planner_hz,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        display=args.display,
        device=args.device,
    )


if __name__ == "__main__":
    main()

