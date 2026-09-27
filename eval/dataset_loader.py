import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Tuple, Union
import cv2
import numpy as np
from PIL import Image


VISDRONE_CATEGORY_NAMES = {
    0: "ignored_regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}


@dataclass
class GroundTruthBox:
    """Represents a single ground-truth object detection / track annotation."""
    target_id: int
    bbox_pixel: List[float]  # [x1, y1, x2, y2]
    width: float
    height: float
    score: int = 1
    category: int = 4
    category_name: str = "car"
    truncation: int = 0
    occlusion: int = 0

    @property
    def center_pixel(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_pixel
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class FrameItem:
    """Represents a single frame and its corresponding ground-truth annotations."""
    frame_idx: int
    image_path: Optional[str] = None
    _image_bgr: Optional[np.ndarray] = None
    gt_boxes: List[GroundTruthBox] = field(default_factory=list)

    def get_bgr(self) -> Optional[np.ndarray]:
        if self._image_bgr is not None:
            return self._image_bgr
        if self.image_path and os.path.exists(self.image_path):
            img = cv2.imread(self.image_path)
            if img is None:
                try:
                    img = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    pass
            if img is None:
                try:
                    pil_img = Image.open(self.image_path).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
            return img
        return None

    def get_rgb_pil(self) -> Optional[Image.Image]:
        bgr = self.get_bgr()
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        return None


class SequenceDataset:
    """
    Parser and loader for sequential UAV datasets (VisDrone-MOT, UAVDT, or ordered frame directories / videos).
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        sequence_path: str,
        gt_path: Optional[str] = None,
        filter_ignored_categories: bool = True,
    ):
        self.sequence_path = os.path.abspath(sequence_path)
        self.filter_ignored_categories = filter_ignored_categories
        self.is_video = False
        self.frame_paths: List[str] = []
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.annotations_by_frame: Dict[int, List[GroundTruthBox]] = {}

        if not os.path.exists(self.sequence_path):
            raise FileNotFoundError(f"Sequence path does not exist: {self.sequence_path}")

        # Check if input is a video file or folder
        ext = os.path.splitext(self.sequence_path)[1].lower()
        if ext in self.VIDEO_EXTENSIONS:
            self.is_video = True
            self.video_capture = cv2.VideoCapture(self.sequence_path)
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self._discover_image_files()

        # Locate and parse ground-truth
        resolved_gt_path = self._locate_gt_file(gt_path)
        if resolved_gt_path and os.path.exists(resolved_gt_path):
            self._parse_gt_file(resolved_gt_path)
            print(f"[SequenceDataset] Loaded GT from {resolved_gt_path} with annotations for {len(self.annotations_by_frame)} frames.")
        else:
            print(f"[SequenceDataset] No ground-truth annotations found for sequence: {self.sequence_path}")

    def _discover_image_files(self) -> None:
        """Finds all image files in sequence directory or subdirectories like images/ or img1/."""
        candidate_dirs = [
            self.sequence_path,
            os.path.join(self.sequence_path, "images"),
            os.path.join(self.sequence_path, "img1"),
            os.path.join(self.sequence_path, "sequences"),
        ]

        found_files = []
        for cdir in candidate_dirs:
            if os.path.isdir(cdir):
                for fname in os.listdir(cdir):
                    fext = os.path.splitext(fname)[1].lower()
                    if fext in self.IMAGE_EXTENSIONS:
                        found_files.append(os.path.join(cdir, fname))
                if found_files:
                    break

        if not found_files:
            raise ValueError(f"No image files found in {self.sequence_path} or candidate subfolders.")

        def natural_sort_key(s: str):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', os.path.basename(s))]

        found_files.sort(key=natural_sort_key)
        self.frame_paths = found_files
        self.total_frames = len(found_files)

    def _locate_gt_file(self, explicit_gt_path: Optional[str]) -> Optional[str]:
        if explicit_gt_path and os.path.exists(explicit_gt_path):
            return explicit_gt_path

        candidates = [
            os.path.join(self.sequence_path, "gt.txt"),
            os.path.join(self.sequence_path, "gt", "gt.txt"),
            os.path.join(self.sequence_path, "annotations", "gt.txt"),
            os.path.join(os.path.dirname(self.sequence_path), "gt", f"{os.path.basename(self.sequence_path)}.txt"),
            os.path.join(os.path.dirname(self.sequence_path), "annotations", f"{os.path.basename(self.sequence_path)}.txt"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None

    def _parse_gt_file(self, gt_path: str) -> None:
        """
        Parses VisDrone or UAVDT formatted gt.txt file.
        VisDrone format:
        <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        """
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.replace(" ", ",").split(",") if p.strip()]
                if len(parts) < 6:
                    continue

                try:
                    frame_idx = int(float(parts[0]))
                    target_id = int(float(parts[1]))
                    x_left = float(parts[2])
                    y_top = float(parts[3])
                    width = float(parts[4])
                    height = float(parts[5])

                    score = int(float(parts[6])) if len(parts) > 6 else 1
                    category = int(float(parts[7])) if len(parts) > 7 else 4
                    truncation = int(float(parts[8])) if len(parts) > 8 else 0
                    occlusion = int(float(parts[9])) if len(parts) > 9 else 0

                    if self.filter_ignored_categories:
                        # Category 0 = ignored regions in VisDrone
                        if category == 0 or score == 0:
                            continue

                    x1 = round(x_left, 2)
                    y1 = round(y_top, 2)
                    x2 = round(x_left + width, 2)
                    y2 = round(y_top + height, 2)

                    cat_name = VISDRONE_CATEGORY_NAMES.get(category, "vehicle")

                    gt_box = GroundTruthBox(
                        target_id=target_id,
                        bbox_pixel=[x1, y1, x2, y2],
                        width=round(width, 2),
                        height=round(height, 2),
                        score=score,
                        category=category,
                        category_name=cat_name,
                        truncation=truncation,
                        occlusion=occlusion,
                    )

                    if frame_idx not in self.annotations_by_frame:
                        self.annotations_by_frame[frame_idx] = []
                    self.annotations_by_frame[frame_idx].append(gt_box)

                except Exception:
                    continue

    def __len__(self) -> int:
        return self.total_frames

    def get_frame(self, index: int) -> FrameItem:
        """
        Returns FrameItem by zero-based frame index.
        Note: VisDrone gt.txt typically uses 1-based indexing for frame_index.
        """
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Index {index} out of range for sequence with {self.total_frames} frames.")

        gt_1based = self.annotations_by_frame.get(index + 1, [])
        gt_0based = self.annotations_by_frame.get(index, [])
        gt_boxes = gt_1based if gt_1based else gt_0based

        if self.is_video:
            assert self.video_capture is not None
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, bgr = self.video_capture.read()
            if not ret or bgr is None:
                raise RuntimeError(f"Could not read frame {index} from video.")
            return FrameItem(frame_idx=index, image_path=None, _image_bgr=bgr, gt_boxes=gt_boxes)
        else:
            img_path = self.frame_paths[index]
            return FrameItem(frame_idx=index, image_path=img_path, _image_bgr=None, gt_boxes=gt_boxes)

    def __iter__(self) -> Iterator[FrameItem]:
        if self.is_video:
            assert self.video_capture is not None
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            idx = 0
            while True:
                ret, bgr = self.video_capture.read()
                if not ret or bgr is None:
                    break
                gt_1based = self.annotations_by_frame.get(idx + 1, [])
                gt_0based = self.annotations_by_frame.get(idx, [])
                gt_boxes = gt_1based if gt_1based else gt_0based
                yield FrameItem(frame_idx=idx, image_path=None, _image_bgr=bgr, gt_boxes=gt_boxes)
                idx += 1
        else:
            for idx, path in enumerate(self.frame_paths):
                gt_1based = self.annotations_by_frame.get(idx + 1, [])
                gt_0based = self.annotations_by_frame.get(idx, [])
                gt_boxes = gt_1based if gt_1based else gt_0based
                yield FrameItem(frame_idx=idx, image_path=path, _image_bgr=None, gt_boxes=gt_boxes)

    def close(self) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
