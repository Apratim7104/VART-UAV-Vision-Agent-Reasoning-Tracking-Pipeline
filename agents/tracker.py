from enum import Enum
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class KalmanFilter2D:
    """
    2D Linear Kalman Filter for constant-velocity kinematic state tracking.
    State Vector: x = [x, y, w, h, vx, vy, vw, vh]^T
    Measurement Vector: z = [xm, ym, wm, hm]^T
    """

    def __init__(
        self,
        init_bbox_center: Tuple[float, float, float, float],
        std_acc: float = 1.0,
        std_meas: float = 1.0,
    ):
        x, y, w, h = init_bbox_center

        # State vector x (8x1)
        self.x = np.array([[x], [y], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)

        # State Covariance Matrix P (8x8)
        self.P = np.eye(8, dtype=np.float64) * 10.0
        # High uncertainty for unobserved initial velocities
        self.P[4:, 4:] *= 100.0

        # Measurement Matrix H (4x8)
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4, dtype=np.float64)

        # Noise scale parameters
        self.std_acc = std_acc
        self.std_meas = std_meas

        # Measurement Noise Covariance Matrix R (4x4)
        self.R = np.eye(4, dtype=np.float64) * (self.std_meas**2)

    def predict(self, dt: float) -> np.ndarray:
        """Projects state estimate and error covariance forward by dt seconds."""
        # State Transition Matrix F (8x8)
        F = np.eye(8, dtype=np.float64)
        for i in range(4):
            F[i, i + 4] = dt

        # Process Noise Covariance Matrix Q (8x8)
        # Discrete continuous-white-noise acceleration model
        Q = np.zeros((8, 8), dtype=np.float64)
        q_pos = (0.25 * (dt**4)) * (self.std_acc**2)
        q_vel = (dt**2) * (self.std_acc**2)
        q_pos_vel = (0.5 * (dt**3)) * (self.std_acc**2)

        for i in range(4):
            Q[i, i] = q_pos
            Q[i + 4, i + 4] = q_vel
            Q[i, i + 4] = q_pos_vel
            Q[i + 4, i] = q_pos_vel

        # A Priori State Prediction: x_k^- = F * x_{k-1}
        self.x = F @ self.x

        # A Priori Covariance Projection: P_k^- = F * P_{k-1} * F^T + Q
        self.P = (F @ self.P @ F.T) + Q

        return self.x

    def update(self, z: np.ndarray) -> None:
        """Corrects the predicted state estimate with new measurement observation z."""
        z = np.reshape(z, (4, 1))

        # Innovation Residual: y_k = z_k - H * x_k^-
        y = z - (self.H @ self.x)

        # Innovation Covariance: S_k = H * P_k^- * H^T + R
        S = (self.H @ self.P @ self.H.T) + self.R

        # Optimal Kalman Gain: K_k = P_k^- * H^T * S_k^-1
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # A Posteriori State Correction: x_k = x_k^- + K_k * y_k
        self.x = self.x + (K @ y)

        # A Posteriori Covariance Correction: P_k = (I - K_k * H) * P_k^-
        I = np.eye(8, dtype=np.float64)
        self.P = (I - (K @ self.H)) @ self.P


class Track:
    """Wrapper holding persistent ID metadata, state status, and Kalman filter instance."""

    def __init__(self, track_id: int, label: str, init_bbox_pixel: List[float]):
        self.track_id = track_id
        self.label = label
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.misses = 0
        self.age = 1

        # Convert [x1, y1, x2, y2] -> [x_center, y_center, w, h]
        x1, y1, x2, y2 = init_bbox_pixel
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        xc = x1 + (w / 2.0)
        yc = y1 + (h / 2.0)

        self.kf = KalmanFilter2D(init_bbox_center=(xc, yc, w, h))

    def predict(self, dt: float) -> List[float]:
        """Predicts filter state and returns estimated [x1, y1, x2, y2] bounding box."""
        x_pred = self.kf.predict(dt)
        return self._state_to_bbox(x_pred)

    def update(self, bbox_pixel: List[float]) -> None:
        """Updates filter state with measurement observation."""
        x1, y1, x2, y2 = bbox_pixel
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        xc = x1 + (w / 2.0)
        yc = y1 + (h / 2.0)

        z = np.array([xc, yc, w, h], dtype=np.float64)
        self.kf.update(z)

        self.hits += 1
        self.misses = 0

    def mark_missed(self) -> None:
        """Increments missed frame counter."""
        self.misses += 1

    def _state_to_bbox(self, x_vec: np.ndarray) -> List[float]:
        xc, yc, w, h = x_vec[:4, 0]
        x1 = xc - (w / 2.0)
        y1 = yc - (h / 2.0)
        x2 = xc + (w / 2.0)
        y2 = yc + (h / 2.0)
        return [float(x1), float(y1), float(x2), float(y2)]

    def get_current_bbox(self) -> List[float]:
        return self._state_to_bbox(self.kf.x)

    def get_velocity(self) -> Tuple[float, float]:
        """Returns instantaneous velocities (vx, vy) in pixels/sec."""
        return float(self.kf.x[4, 0]), float(self.kf.x[5, 0])


class KalmanTracker:
    """
    Central Multi-Object Tracker. Handles bipartite matching between predicted
    tracks and incoming VLM detections, track promotion, and deletion.
    """

    def __init__(self, iou_threshold: float = 0.3, n_min: int = 3, m_max: int = 5):
        self.iou_threshold = iou_threshold
        self.n_min = n_min
        self.m_max = m_max
        self.tracks: List[Track] = []
        self._next_id = 101

    @staticmethod
    def compute_iou(boxA: List[float], boxB: List[float]) -> float:
        """Computes Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(1e-5, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1e-5, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        return interArea / float(boxAArea + boxBArea - interArea)

    def update(
        self,
        perception_output: Dict[str, Any],
        dt: float = 0.066  # ~15 FPS frame step
    ) -> List[Dict[str, Any]]:
        """
        Processes new detections, runs state prediction, updates matches,
        and returns confirmed tracks formatted for PlannerAgent.
        """
        detections = perception_output.get("detections", [])
        frame_size = perception_output.get("frame_size", {"width": 1280, "height": 720})
        w_img, h_img = frame_size["width"], frame_size["height"]

        # Step 1: Predict new state positions for all active tracks
        predicted_boxes = []
        for track in self.tracks:
            track.age += 1
            predicted_boxes.append(track.predict(dt))

        # Step 2: Compute Cost Matrix (1.0 - IoU)
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        cost_matrix = np.ones((num_tracks, num_dets), dtype=np.float32)
        for i, track_box in enumerate(predicted_boxes):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1.0 - self.compute_iou(track_box, det["bbox_pixel"])

        # Step 3: Hungarian Algorithm Bipartite Matching
        if num_tracks > 0 and num_dets > 0:
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
        else:
            track_indices, det_indices = np.array([], dtype=int), np.array([], dtype=int)

        # Step 4: Separate Matches, Unmatched Detections, Unmatched Tracks
        matched_track_ids = set()
        matched_det_ids = set()

        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] <= (1.0 - self.iou_threshold):
                matched_track_ids.add(t_idx)
                matched_det_ids.add(d_idx)

                # Correct Kalman Filter State
                self.tracks[t_idx].update(detections[d_idx]["bbox_pixel"])

        # Step 5: Handle Unmatched Tracks
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_ids:
                track.mark_missed()

        # Step 6: Handle Unmatched Detections (Spawn New Tentative Tracks)
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_ids:
                new_track = Track(
                    track_id=self._next_id,
                    label=det["label"],
                    init_bbox_pixel=det["bbox_pixel"]
                )
                self._next_id += 1
                self.tracks.append(new_track)

        # Step 7: Manage Track Lifecycle Promotions & Deletions
        active_confirmed_tracks = []
        for track in self.tracks:
            # Promote TENTATIVE -> CONFIRMED
            if track.state == TrackState.TENTATIVE and track.hits >= self.n_min:
                track.state = TrackState.CONFIRMED

            # Mark DELETED if missing too long
            if track.misses > self.m_max:
                track.state = TrackState.DELETED

            # Export active confirmed tracks for downstream Planner Agent
            if track.state == TrackState.CONFIRMED:
                bbox = track.get_current_bbox()
                vx, vy = track.get_velocity()
                xc, yc = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

                offset_x = (xc - (w_img / 2.0)) / (w_img / 2.0)
                offset_y = (yc - (h_img / 2.0)) / (h_img / 2.0)

                active_confirmed_tracks.append({
                    "track_id": track.track_id,
                    "label": track.label,
                    "bbox_pixel": [round(c, 2) for c in bbox],
                    "center_pixel": [round(xc, 2), round(yc, 2)],
                    "center_offset_normalized": [round(offset_x, 4), round(offset_y, 4)],
                    "velocity_pixel_s": [round(vx, 2), round(vy, 2)],
                    "age_frames": track.age,
                    "missed_frames": track.misses,
                })

        # Purge DELETED tracks from internal memory
        self.tracks = [t for t in self.tracks if t.state != TrackState.DELETED]

        return active_confirmed_tracks


if __name__ == "__main__":
    # Test pipeline integration using mock perception output
    mock_perception_frame = {
        "frame_size": {"width": 1280, "height": 720},
        "detections": [
            {"label": "car", "bbox_pixel": [500.0, 300.0, 600.0, 400.0]},
            {"label": "car", "bbox_pixel": [100.0, 100.0, 200.0, 200.0]}
        ]
    }

    tracker = KalmanTracker(n_min=1)  # Set n_min=1 for immediate confirmation in test
    confirmed = tracker.update(mock_perception_frame, dt=0.066)

    print("\n--- Kalman Tracker Test Output ---")
    print(f"Confirmed Active Tracks Count: {len(confirmed)}")
    if confirmed:
        print("Sample Tracked Target State:")
        print(confirmed[0])