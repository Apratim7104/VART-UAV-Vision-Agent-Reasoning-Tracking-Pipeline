# UAV Florence-2 Multi-Agent Autonomous Flight Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-blueviolet)](https://ollama.com/)
[![Florence-2](https://img.shields.io/badge/Florence--2-base-orange)](https://huggingface.co/microsoft/Florence-2-base)

> **A four-layer multi-agent architecture for autonomous UAV object detection, Kalman-filter multi-object tracking, LLM-driven tactical planning, and closed-loop flight control — benchmarked on VisDrone2019-MOT and UAVDT datasets.**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Single-Sequence Inference](#single-sequence-inference)
  - [VisDrone / UAVDT Benchmark Evaluation](#visdrone--uavdt-benchmark-evaluation)
  - [Batch Evaluation with PDF Report](#batch-evaluation-with-pdf-report)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository implements a **real-time multi-agent pipeline** for aerial surveillance and autonomous UAV guidance. It chains four specialized agents into a coherent perception-to-control loop:

| Layer | Agent | Model / Algorithm |
|-------|-------|-------------------|
| 1 — Perception | `PerceptionAgent` | Microsoft **Florence-2-base** (VLM) |
| 2 — Tracking | `KalmanTracker` | 8-State **Kalman Filter** + Hungarian bipartite matching |
| 3 — Planning | `PlannerAgent` | **Qwen2.5-1.5B** via Ollama (local CPU LLM) |
| 4 — Control | `ControllerAgent` | Proportional **4-DOF velocity** setpoint generator |

The output is an annotated evaluation video with an on-screen HUD displaying telemetry, tracking overlays, planner decisions, and 4-DOF control setpoints.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  UAV MULTI-AGENT PIPELINE                           │
│                                                                     │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────────┐  │
│  │ SequenceData │───►│ PerceptionAgent│───►│   KalmanTracker     │  │
│  │ (VisDrone /  │    │  Florence-2    │    │  8-state KF +       │  │
│  │  UAVDT /     │    │  Object Detect │    │  Hungarian Matching │  │
│  │  Video)      │    └────────────────┘    └──────────┬──────────┘  │
│  └──────────────┘         ▲                           │             │
│                           │ PIL Image                 │ Tracks      │
│                           │                           ▼             │
│                    ┌──────┴───────┐          ┌────────────────┐     │
│                    │  Video HUD   │◄─────────│ PlannerAgent   │     │
│                    │  Renderer    │ Setpoints │  Qwen2.5-1.5B │     │
│                    │  (OpenCV)    │           │  via Ollama    │     │
│                    └──────┬───────┘          └────────┬───────┘     │
│                           │                           │ Plan        │
│                           │                           ▼             │
│                    ┌──────┴───────┐          ┌────────────────┐     │
│                    │ Output Video │          │ControllerAgent │     │
│                    │  (.mp4)      │          │  4-DOF P-ctrl  │     │
│                    └─────────────┘          └────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

**Control Actions (Planner FSM):**

| State | Trigger | Behaviour |
|-------|---------|-----------|
| `SEARCH` | No confirmed tracks | Slow yaw scan |
| `CENTER` | Target offset > 0.25 (normalized) | Yaw + altitude correction |
| `TRACK` | Target offset ≤ 0.25 | Forward advance + fine correction |
| `HOLD` | Initialization / error recovery | Zero velocity hover |

---

## Features

- 🔭 **Zero-shot object detection** using Microsoft Florence-2 — no dataset fine-tuning needed
- 🎯 **Multi-object tracking** with a continuous 8-state Kalman filter (position, size, velocity) and Hungarian algorithm for track-detection association
- 🧠 **On-device LLM planning** via Ollama (Qwen2.5-1.5B) — no cloud API, fully offline
- 🔄 **Deterministic rule-based fallback** planner for graceful degradation when Ollama is unavailable
- 📊 **Quantitative benchmarking** on VisDrone2019-MOT and UAVDT with mIoU, ID-Switches, Fragmentations, Mostly-Tracked, and Mostly-Lost ratios
- 📹 **HUD-annotated output video** with bounding boxes, velocity vectors, targeting reticle, and telemetry overlay
- 📄 **PDF batch report generation** across multiple sequences with checkpointing/resume support
- ⚡ **GPU-accelerated inference** (CUDA) with automatic CPU fallback
- 🧪 **Pytest integration tests** covering all pipeline layers

---

## Project Structure

```
uav_florence2_agent/
│
├── agents/                      # Core agent modules
│   ├── __init__.py
│   ├── perception.py            # PerceptionAgent — Florence-2 VLM wrapper
│   ├── tracker.py               # KalmanTracker — 8-state KF + Hungarian matching
│   ├── planner.py               # PlannerAgent — Ollama LLM tactical planner
│   └── controller.py            # ControllerAgent — 4-DOF proportional controller
│
├── eval/                        # Evaluation framework
│   ├── __init__.py
│   ├── dataset_loader.py        # SequenceDataset — VisDrone / UAVDT / video loader
│   ├── metrics.py               # BenchmarkEvaluator — mIoU, IDSW, MT/ML metrics
│   ├── evaluate_visdrone.py     # Single-sequence evaluation runner
│   └── batch_evaluate.py        # Multi-sequence batch runner with PDF reports
│
├── tests/                       # Integration test suite
│   └── test_pipeline_integration.py
│
├── data/                        # Data directory (datasets not committed)
│   ├── sample_sequence/         # Minimal 5-frame sample for quick testing
│   │   ├── images/              # 0000001.jpg … 0000005.jpg
│   │   └── gt.txt               # VisDrone-format ground-truth
│   └── batch_dataset/           # Multi-sequence sample for batch testing
│       ├── seq1/
│       └── seq2/
│
├── detect_objects.py            # Standalone Florence-2 detection demo script
├── run_pipeline.py              # Main pipeline entry point (single sequence)
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| Python | 3.10 | 3.11 / 3.12 also tested |
| PyTorch | 2.1.0 | GPU: install CUDA-enabled wheel separately |
| CUDA (optional) | 11.8 / 12.x | Strongly recommended for Florence-2 |
| Ollama | 0.2.0 | Must be running as a daemon for `PlannerAgent` |
| RAM | 8 GB | 16 GB recommended for GPU inference |
| VRAM | 4 GB | Florence-2-base in fp16 |

---

## Installation

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/uav_florence2_agent.git
cd uav_florence2_agent
```

### 2 — Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

**For GPU acceleration** (CUDA 12.x), install a CUDA-enabled PyTorch wheel first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4 — Install and start Ollama, pull the planner model

```bash
# Install Ollama — https://ollama.com/download
# Then pull the planner model:
ollama pull qwen2.5:1.5b

# Verify Ollama is running (default port 11434)
ollama serve
```

> **Note:** If Ollama is unavailable, the `PlannerAgent` automatically falls back to its deterministic rule-based engine.

### 5 — Florence-2 model weights

Weights are downloaded automatically from HuggingFace Hub on first run:

```
microsoft/Florence-2-base   (~460 MB)
```

Ensure you have internet access on the first run, or pre-cache with:

```bash
python -c "from transformers import AutoModelForCausalLM, AutoProcessor; AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True); print('Cached.')"
```

---

## Quick Start

Run the pipeline on the provided 5-frame sample sequence:

```bash
python run_pipeline.py \
  --input data/sample_sequence \
  --output output_evaluation.mp4 \
  --planner-hz 2.5
```

The annotated video will be saved to `output_evaluation.mp4`.

---

## Usage

### Single-Sequence Inference

```bash
python run_pipeline.py \
  --input  /path/to/sequence_dir \   # folder with images/ subfolder, or a .mp4 file
  --output output_evaluation.mp4 \   # output annotated video path
  --planner-hz 2.5 \                 # LLM planner invocation rate (Hz)
  --max-frames 200 \                 # optional frame cap
  --display \                        # optional: show live OpenCV window
  --device cuda                      # or 'cpu' — auto-detected if omitted
```

**Supported input formats:**

| Format | Example |
|--------|---------|
| VisDrone-MOT sequence folder | `data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v` |
| UAVDT sequence folder | `data/UAVDT/UAV-benchmark-M/M0101` |
| Generic image folder | `my_drone_frames/` (sorted by filename) |
| Video file | `flight_recording.mp4` |

---

### VisDrone / UAVDT Benchmark Evaluation

Runs end-to-end quantitative evaluation on a single sequence:

```bash
python eval/evaluate_visdrone.py \
  --sequence-dir data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
  --output-dir   eval_output/uav0000086 \
  --gt-path      data/VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt \
  --planner-hz   2.5 \
  --device       cuda
```

Results are written to `eval_output/uav0000086/eval_results.json` and printed to stdout as a formatted benchmark table.

---

### Batch Evaluation with PDF Report

Runs across all sequences in a dataset root with automatic checkpointing:

```bash
python eval/batch_evaluate.py \
  --dataset-root data/VisDrone2019-MOT-val/sequences \
  --output-dir   batch_eval_results \
  --planner-hz   2.5 \
  --device       cuda

# Resume an interrupted run:
python eval/batch_evaluate.py \
  --dataset-root data/VisDrone2019-MOT-val/sequences \
  --output-dir   batch_eval_results \
  --resume
```

A PDF benchmark report (`batch_report_final.pdf`) is generated at completion.

---

### Standalone Detection Demo

Run Florence-2 object detection on any single image:

```bash
# Place your image at the project root as test_image.png (or edit the path in detect_objects.py)
python detect_objects.py
# → saves annotated_result.png
```

---

## Dataset Setup

### VisDrone2019-MOT

1. Download from the [VisDrone official page](https://github.com/VisDrone/VisDrone-Dataset)
2. Extract and place under `data/`:

```
data/
└── VisDrone2019-MOT-val/
    └── VisDrone2019-MOT-val/
        ├── annotations/          ← *.txt ground-truth files
        └── sequences/            ← sequence folders with images/
```

### UAVDT

1. Download from the [UAVDT benchmark page](https://sites.google.com/site/daviddo0323/projects/uavdt)
2. Extract under `data/`:

```
data/
└── UAVDT/
    └── UAV-benchmark-M/
        ├── M0101/
        │   └── img000001.jpg …
        └── …
```

### Custom Sequence

Any folder containing sorted image files (`.jpg`, `.png`, etc.) works as input. Optionally place a `gt.txt` in VisDrone MOT format alongside:

```
<frame_id>, <target_id>, <x_left>, <y_top>, <width>, <height>, <score>, <category>, <truncation>, <occlusion>
```

---

## Configuration

Key parameters that can be adjusted at the CLI or in agent constructors:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--planner-hz` | `2.5` | Maximum LLM invocation rate (Hz) |
| `--device` | auto | `cuda` or `cpu` |
| `iou_threshold` | `0.3` | Minimum IoU for track-detection match |
| `n_min` | `2` | Frames before track is promoted to CONFIRMED |
| `m_max` | `5` | Max missed frames before track is deleted |
| `k_p_yaw` | `1.2` | Proportional gain for yaw control |
| `k_p_alt` | `0.8` | Proportional gain for altitude control |
| `max_vel_mps` | `2.0` | Maximum velocity setpoint (m/s) |
| `max_yaw_rate_rads` | `0.5` | Maximum yaw rate setpoint (rad/s) |
| `model_id` (Florence-2) | `microsoft/Florence-2-base` | Can be swapped to `Florence-2-large` |
| `model_name` (Planner) | `qwen2.5:1.5b` | Any Ollama-compatible model |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **mIoU** | Mean Intersection-over-Union between predicted and ground-truth bounding boxes across matched frames |
| **IDSW** | Identity Switches — number of times a GT target is assigned a different predicted ID |
| **Frag** | Fragmentations — number of trajectory interruptions in tracking continuity |
| **MT** | Mostly Tracked — fraction of GT trajectories tracked ≥ 80 % of their lifetime |
| **ML** | Mostly Lost — fraction of GT trajectories tracked ≤ 20 % of their lifetime |
| **Perception Latency** | Florence-2 end-to-end inference time per frame (ms) |
| **Planner Latency** | Ollama LLM decision time per invocation (ms) |
| **System FPS** | End-to-end wall-clock throughput |

---

## Testing

Run the integration test suite (no GPU required — uses synthetic frames):

```bash
pytest tests/ -v
```

The tests cover:

- `test_dataset_loader` — VisDrone sequence parsing and GT loading
- `test_kalman_tracker_and_controller` — Track lifecycle, ID assignment, and 4-DOF setpoint generation
- `test_metrics_evaluator` — mIoU computation, action distribution, and latency recording

---

## Roadmap

- [ ] Florence-2-large variant support
- [ ] Deep-SORT Re-ID feature embedding for improved track continuity
- [ ] ROS2 node wrapper for real UAV deployment
- [ ] ONNX export for Florence-2 for edge inference
- [ ] DeepStream / TensorRT integration for high-FPS embedded systems
- [ ] Web dashboard for live telemetry streaming

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{uav_florence2_agent,
  title   = {UAV Florence-2 Multi-Agent Autonomous Flight Pipeline},
  author  = {Apratim},
  year    = {2026},
  url     = {https://github.com/<your-username>/uav_florence2_agent},
}
```

**Upstream models and datasets used:**

```bibtex
@article{xiao2024florence2,
  title   = {Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
  author  = {Xiao, Bin and others},
  journal = {arXiv preprint arXiv:2311.06242},
  year    = {2024}
}

@inproceedings{zhu2018visdrone,
  title     = {Vision Meets Drones: A Challenge},
  author    = {Zhu, Pengfei and others},
  booktitle = {arXiv preprint arXiv:1804.07437},
  year      = {2018}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

> **Third-party licenses:** Florence-2 weights are subject to the [Microsoft Research License Agreement](https://huggingface.co/microsoft/Florence-2-base). VisDrone and UAVDT datasets are subject to their respective academic research licenses.
