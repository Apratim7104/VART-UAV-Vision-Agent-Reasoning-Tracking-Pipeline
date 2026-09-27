import os
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return float(interArea / (boxAArea + boxBArea - interArea))


@dataclass
class GTTrajectoryInfo:
    first_frame: int = 0
    last_frame: int = 0
    total_frames: int = 0
    tracked_frames: int = 0
    last_tracked_frame: int = -1
    last_matched_pred_id: Optional[int] = None


@dataclass
class EvaluationReport:
    """Quantitative results container for UAV Multi-Agent Pipeline Benchmark."""
    total_frames: int = 0
    total_time_s: float = 0.0
    average_fps: float = 0.0

    # Perception
    perception_mean_latency_ms: float = 0.0
    perception_median_latency_ms: float = 0.0
    perception_std_latency_ms: float = 0.0
    perception_min_latency_ms: float = 0.0
    perception_max_latency_ms: float = 0.0
    total_detections: int = 0

    # Tracking
    mean_iou: float = 0.0
    id_switches: int = 0
    fragmentations: int = 0
    mostly_tracked_ratio: float = 0.0
    mostly_lost_ratio: float = 0.0
    total_gt_targets: int = 0
    total_pred_tracks: int = 0

    # Planner
    planner_mean_latency_ms: float = 0.0
    planner_call_count: int = 0
    action_distribution: Dict[str, float] = field(default_factory=dict)
    action_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_frames": self.total_frames,
                "total_duration_s": round(self.total_time_s, 2),
                "system_average_fps": round(self.average_fps, 2),
            },
            "perception": {
                "mean_latency_ms": round(self.perception_mean_latency_ms, 2),
                "median_latency_ms": round(self.perception_median_latency_ms, 2),
                "std_latency_ms": round(self.perception_std_latency_ms, 2),
                "min_latency_ms": round(self.perception_min_latency_ms, 2),
                "max_latency_ms": round(self.perception_max_latency_ms, 2),
                "total_detections": self.total_detections,
            },
            "tracking": {
                "mean_iou": round(self.mean_iou, 4),
                "id_switches": self.id_switches,
                "fragmentations": self.fragmentations,
                "mostly_tracked_ratio": round(self.mostly_tracked_ratio, 4),
                "mostly_lost_ratio": round(self.mostly_lost_ratio, 4),
                "total_gt_targets": self.total_gt_targets,
                "unique_predicted_tracks": self.total_pred_tracks,
            },
            "planner": {
                "mean_latency_ms": round(self.planner_mean_latency_ms, 2),
                "total_calls": self.planner_call_count,
                "action_distribution_pct": {
                    k: round(v * 100.0, 2) for k, v in self.action_distribution.items()
                },
                "action_counts": self.action_counts,
            }
        }


class BenchmarkEvaluator:
    """
    Evaluates tracking, perception, and planner agent performance over sequential frames.
    """

    def __init__(self, match_iou_thresh: float = 0.3):
        self.match_iou_thresh = match_iou_thresh

        # Timings
        self.frame_count = 0
        self.start_wall_time = time.time()
        self.perception_latencies: List[float] = []
        self.planner_latencies: List[float] = []
        self.total_detections_count = 0

        # Action history
        self.action_history: List[str] = []

        # Tracking metrics internal state
        self.iou_scores: List[float] = []
        self.id_switches = 0
        self.fragmentations = 0
        self.unique_pred_ids: Set[int] = set()

        # GT target trajectories: gt_id -> GTTrajectoryInfo
        self.gt_trajectories: Dict[int, GTTrajectoryInfo] = {}

    def step(
        self,
        frame_idx: int,
        perception_latency_ms: float,
        num_detections: int,
        predicted_tracks: List[Dict[str, Any]],
        ground_truth_boxes: Optional[List[Any]] = None,
        planner_output: Optional[Dict[str, Any]] = None,
        planner_latency_ms: Optional[float] = None,
    ) -> None:
        """
        Records per-frame metrics for perception, tracking, and planner layers.
        """
        self.frame_count += 1
        self.perception_latencies.append(perception_latency_ms)
        self.total_detections_count += num_detections

        # Record planner metrics
        if planner_output is not None:
            action = str(planner_output.get("action", "HOLD")).upper().strip()
            self.action_history.append(action)

        if planner_latency_ms is not None and planner_latency_ms > 0:
            self.planner_latencies.append(planner_latency_ms)

        # Track unique predicted IDs
        for p in predicted_tracks:
            pid = p.get("track_id")
            if pid is not None:
                self.unique_pred_ids.add(pid)

        # Evaluate tracking if ground-truth is available
        if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
            self._evaluate_frame_tracking(frame_idx, predicted_tracks, ground_truth_boxes)

    def _evaluate_frame_tracking(
        self,
        frame_idx: int,
        predicted_tracks: List[Dict[str, Any]],
        gt_boxes: List[Any]
    ) -> None:
        num_gt = len(gt_boxes)
        num_pred = len(predicted_tracks)

        # Update GT trajectory lifetimes
        current_frame_gt_ids = set()
        for gt in gt_boxes:
            gt_id = getattr(gt, "target_id", None) if hasattr(gt, "target_id") else gt.get("target_id")
            if gt_id is None:
                continue
            current_frame_gt_ids.add(gt_id)
            if gt_id not in self.gt_trajectories:
                self.gt_trajectories[gt_id] = GTTrajectoryInfo(
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    total_frames=1,
                    tracked_frames=0,
                    last_tracked_frame=-1,
                    last_matched_pred_id=None,
                )
            else:
                info = self.gt_trajectories[gt_id]
                info.last_frame = frame_idx
                info.total_frames += 1

        if num_gt == 0 or num_pred == 0:
            return

        # Build cost matrix (1.0 - IoU)
        cost_matrix = np.ones((num_gt, num_pred), dtype=np.float32)
        iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)

        for i, gt in enumerate(gt_boxes):
            gt_bbox = getattr(gt, "bbox_pixel", None) if hasattr(gt, "bbox_pixel") else gt.get("bbox_pixel")
            for j, pred in enumerate(predicted_tracks):
                pred_bbox = pred.get("bbox_pixel")
                if gt_bbox and pred_bbox:
                    iou = compute_iou(gt_bbox, pred_bbox)
                    iou_matrix[i, j] = iou
                    cost_matrix[i, j] = 1.0 - iou

        gt_ind, pred_ind = linear_sum_assignment(cost_matrix)

        matched_gt_ids = set()
        for g_idx, p_idx in zip(gt_ind, pred_ind):
            iou_val = iou_matrix[g_idx, p_idx]
            if iou_val >= self.match_iou_thresh:
                self.iou_scores.append(iou_val)

                gt_item = gt_boxes[g_idx]
                gt_id = getattr(gt_item, "target_id", None) if hasattr(gt_item, "target_id") else gt_item.get("target_id")
                pred_id = predicted_tracks[p_idx].get("track_id")

                if gt_id is not None:
                    matched_gt_ids.add(gt_id)
                    info = self.gt_trajectories[gt_id]
                    info.tracked_frames += 1

                    # Check for ID Switch
                    if info.last_matched_pred_id is not None and info.last_matched_pred_id != pred_id:
                        self.id_switches += 1

                    # Check for Fragmentation (interruption in consecutive tracking)
                    if info.last_tracked_frame != -1 and (frame_idx - info.last_tracked_frame > 1):
                        self.fragmentations += 1

                    info.last_tracked_frame = frame_idx
                    info.last_matched_pred_id = pred_id

    def finalize(self) -> EvaluationReport:
        """Computes summary metrics and returns EvaluationReport."""
        total_time = max(0.001, time.time() - self.start_wall_time)
        fps = self.frame_count / total_time if total_time > 0 else 0.0

        # Perception latencies
        p_lat = self.perception_latencies if self.perception_latencies else [0.0]
        mean_p = float(np.mean(p_lat))
        med_p = float(np.median(p_lat))
        std_p = float(np.std(p_lat))
        min_p = float(np.min(p_lat))
        max_p = float(np.max(p_lat))

        # Tracking metrics
        m_iou = float(np.mean(self.iou_scores)) if self.iou_scores else 0.0

        # Mostly Tracked / Mostly Lost ratios
        mt_count = 0
        ml_count = 0
        total_gt = len(self.gt_trajectories)

        for gt_id, info in self.gt_trajectories.items():
            if info.total_frames > 0:
                coverage = info.tracked_frames / float(info.total_frames)
                if coverage >= 0.8:
                    mt_count += 1
                elif coverage <= 0.2:
                    ml_count += 1

        mt_ratio = (mt_count / float(total_gt)) if total_gt > 0 else 0.0
        ml_ratio = (ml_count / float(total_gt)) if total_gt > 0 else 0.0

        # Planner distribution and latency
        plan_lat = self.planner_latencies if self.planner_latencies else [0.0]
        mean_plan = float(np.mean(plan_lat))

        action_counts = {}
        for act in self.action_history:
            action_counts[act] = action_counts.get(act, 0) + 1

        total_actions = max(1, len(self.action_history))
        action_dist = {k: v / float(total_actions) for k, v in action_counts.items()}

        report = EvaluationReport(
            total_frames=self.frame_count,
            total_time_s=total_time,
            average_fps=fps,
            perception_mean_latency_ms=mean_p,
            perception_median_latency_ms=med_p,
            perception_std_latency_ms=std_p,
            perception_min_latency_ms=min_p,
            perception_max_latency_ms=max_p,
            total_detections=self.total_detections_count,
            mean_iou=m_iou,
            id_switches=self.id_switches,
            fragmentations=self.fragmentations,
            mostly_tracked_ratio=mt_ratio,
            mostly_lost_ratio=ml_ratio,
            total_gt_targets=total_gt,
            total_pred_tracks=len(self.unique_pred_ids),
            planner_mean_latency_ms=mean_plan,
            planner_call_count=len(self.planner_latencies),
            action_distribution=action_dist,
            action_counts=action_counts,
        )

        return report

    def save_json(self, output_path: str = "eval_results.json") -> str:
        """Saves evaluation summary to JSON file."""
        report = self.finalize()
        out_dict = report.to_dict()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_dict, f, indent=2)

        return output_path

