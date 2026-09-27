import os
import sys
import shutil
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.perception import PerceptionAgent
from agents.tracker import KalmanTracker
from agents.planner import PlannerAgent
from agents.controller import ControllerAgent
from eval.dataset_loader import SequenceDataset
from eval.metrics import BenchmarkEvaluator, compute_iou
from run_pipeline import run_pipeline
from eval.evaluate_visdrone import evaluate_sequence


def create_mock_sequence(base_dir: str, num_frames: int = 5):
    """Creates a mock VisDrone-style sequence with images and gt.txt."""
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    gt_lines = []
    w, h = 640, 480

    for i in range(1, num_frames + 1):
        # Create a simple frame with two moving vehicle-like blocks
        img = np.full((h, w, 3), 40, dtype=np.uint8)

        # Vehicle 1 (moves right)
        x1 = 200 + i * 15
        y1 = 180 + i * 3
        bw1, bh1 = 120, 60
        cv2.rectangle(img, (x1, y1), (x1 + bw1, y1 + bh1), (0, 0, 200), -1)
        cv2.putText(img, "Car 1", (x1 + 5, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        gt_lines.append(f"{i}, 1, {x1}, {y1}, {bw1}, {bh1}, 1, 4, 0, 0\n")

        # Vehicle 2 (moves left)
        x2 = 450 - i * 10
        y2 = 300 - i * 2
        bw2, bh2 = 100, 50
        cv2.rectangle(img, (x2, y2), (x2 + bw2, y2 + bh2), (200, 0, 0), -1)
        cv2.putText(img, "Car 2", (x2 + 5, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        gt_lines.append(f"{i}, 2, {x2}, {y2}, {bw2}, {bh2}, 1, 4, 0, 0\n")

        img_path = os.path.join(images_dir, f"{i:07d}.jpg")
        cv2.imwrite(img_path, img)

    gt_file_path = os.path.join(base_dir, "gt.txt")
    with open(gt_file_path, "w", encoding="utf-8") as f:
        f.writelines(gt_lines)

    return base_dir


def test_dataset_loader(tmp_path):
    seq_dir = os.path.join(str(tmp_path), "test_seq")
    create_mock_sequence(seq_dir, num_frames=5)

    dataset = SequenceDataset(seq_dir)
    assert len(dataset) == 5

    frame0 = dataset.get_frame(0)
    assert frame0.frame_idx == 0
    assert len(frame0.gt_boxes) == 2
    bgr = frame0.get_bgr()
    assert bgr is not None
    assert bgr.shape == (480, 640, 3)

    gt1 = frame0.gt_boxes[0]
    assert gt1.target_id == 1
    assert gt1.category_name == "car"


def test_kalman_tracker_and_controller():
    tracker = KalmanTracker(iou_threshold=0.3, n_min=2, m_max=3)
    controller = ControllerAgent()

    # Frame 1 detection
    mock_perception_1 = {
        "frame_size": {"width": 640, "height": 480},
        "detections": [
            {
                "label": "car",
                "bbox_pixel": [200.0, 150.0, 300.0, 220.0],
                "center_offset_normalized": [0.1, -0.2],
            }
        ],
    }
    # First update: track is tentative
    tracks_f1 = tracker.update(mock_perception_1, dt=0.066)
    assert len(tracks_f1) == 0  # n_min=2, so not confirmed yet

    # Frame 2 detection: should promote to CONFIRMED
    mock_perception_2 = {
        "frame_size": {"width": 640, "height": 480},
        "detections": [
            {
                "label": "car",
                "bbox_pixel": [210.0, 152.0, 310.0, 222.0],
                "center_offset_normalized": [0.12, -0.19],
            }
        ],
    }
    tracks_f2 = tracker.update(mock_perception_2, dt=0.066)
    assert len(tracks_f2) == 1
    confirmed_track = tracks_f2[0]
    assert confirmed_track["track_id"] == 101

    # Planner mock plan
    planner_plan = {
        "action": "TRACK",
        "primary_target_id": 101,
        "reasoning": "Target centered, tracking.",
    }

    setpoints = controller.compute_setpoints(planner_plan, primary_target=confirmed_track)
    assert setpoints["executed_action"] == "TRACK"
    sp = setpoints["setpoints"]
    assert "vx_m_s" in sp
    assert "vz_m_s" in sp
    assert "yaw_rate_rad_s" in sp
    assert sp["vx_m_s"] > 0.0  # forward velocity during TRACK


def test_metrics_evaluator():
    evaluator = BenchmarkEvaluator(match_iou_thresh=0.3)

    # Frame 1
    pred_tracks = [{"track_id": 101, "bbox_pixel": [100.0, 100.0, 200.0, 200.0]}]
    gt_boxes = [{"target_id": 1, "bbox_pixel": [105.0, 105.0, 202.0, 198.0]}]

    evaluator.step(
        frame_idx=1,
        perception_latency_ms=25.0,
        num_detections=1,
        predicted_tracks=pred_tracks,
        ground_truth_boxes=gt_boxes,
        planner_output={"action": "CENTER", "primary_target_id": 101},
        planner_latency_ms=15.0,
    )

    report = evaluator.finalize()
    assert report.total_frames == 1
    assert report.mean_iou > 0.8
    assert report.id_switches == 0
    assert report.perception_mean_latency_ms == 25.0
    assert "CENTER" in report.action_distribution


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        seq_dir = os.path.join(temp_dir, "test_seq")
        create_mock_sequence(seq_dir, num_frames=5)
        print("\n--- Running SequenceDataset Test ---")
        test_dataset_loader(temp_dir)
        print("DatasetLoader test PASSED!")

        print("\n--- Running KalmanTracker & Controller Test ---")
        test_kalman_tracker_and_controller()
        print("KalmanTracker & Controller test PASSED!")

        print("\n--- Running BenchmarkEvaluator Test ---")
        test_metrics_evaluator()
        print("BenchmarkEvaluator test PASSED!")
