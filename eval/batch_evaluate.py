import argparse
import os
import sys
import json
import time
from typing import List, Dict, Any, Optional
from fpdf import FPDF

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.evaluate_visdrone import evaluate_sequence

class PDFReport(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10, "UAV Florence-2 Multi-Agent Benchmark Report", border=False, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf_report(report_path: str, aggregated_metrics: Dict[str, Any], sequence_stats: List[Dict[str, Any]]):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)

    total_seq = aggregated_metrics.get("total_sequences", 0)
    total_frames = aggregated_metrics.get("total_frames", 0)
    avg_fps = aggregated_metrics.get("average_fps", 0.0)
    mean_iou = aggregated_metrics.get("mean_iou", 0.0)
    avg_perc_lat = aggregated_metrics.get("average_perception_latency_ms", 0.0)

    # Summary
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=12)
    
    pdf.cell(0, 8, f"Total Sequences Evaluated: {total_seq}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Total Frames Processed: {total_frames}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Average System Throughput (FPS): {avg_fps:.2f}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Average Perception Latency: {avg_perc_lat:.2f} ms", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Overall Mean Tracking IoU (Task Success): {mean_iou:.4f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Detailed table for each sequence
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, "Sequence Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)

    # Table Header
    col_w = [40, 25, 25, 30, 25, 45]
    headers = ["Sequence", "Frames", "Time (s)", "Perception (ms)", "mIoU", "Mostly Tracked/Lost (%)"]
    
    pdf.set_font("helvetica", "B", 10)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("helvetica", size=9)
    for stat in sequence_stats:
        pdf.cell(col_w[0], 8, stat.get("seq_name", "")[:20], border=1)
        pdf.cell(col_w[1], 8, str(stat.get("frames", 0)), border=1, align="R")
        pdf.cell(col_w[2], 8, f"{stat.get('duration_s', 0):.1f}", border=1, align="R")
        pdf.cell(col_w[3], 8, f"{stat.get('perc_lat_ms', 0):.1f}", border=1, align="R")
        pdf.cell(col_w[4], 8, f"{stat.get('miou', 0):.4f}", border=1, align="R")
        mt = stat.get('mostly_tracked', 0)
        ml = stat.get('mostly_lost', 0)
        pdf.cell(col_w[5], 8, f"MT: {mt:.1f}% / ML: {ml:.1f}%", border=1, align="C")
        pdf.ln()

    pdf.output(report_path)
    print(f"[REPORT] Saved PDF report to {report_path}")

def update_aggregated_metrics(metrics: Dict[str, Any], sequence_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sequence_stats:
        return metrics
    
    total_frames = sum(s.get("frames", 0) for s in sequence_stats)
    total_dur = sum(s.get("duration_s", 0) for s in sequence_stats)
    avg_fps = total_frames / total_dur if total_dur > 0 else 0
    
    avg_miou = sum(s.get("miou", 0) for s in sequence_stats) / len(sequence_stats)
    avg_perc = sum(s.get("perc_lat_ms", 0) for s in sequence_stats) / len(sequence_stats)

    metrics["total_sequences"] = len(sequence_stats)
    metrics["total_frames"] = total_frames
    metrics["total_duration_s"] = total_dur
    metrics["average_fps"] = avg_fps
    metrics["mean_iou"] = avg_miou
    metrics["average_perception_latency_ms"] = avg_perc

    return metrics

def batch_evaluate(
    dataset_root: str,
    output_dir: str,
    resume: bool = False,
    planner_hz: float = 2.5,
    max_frames_per_seq: Optional[int] = None,
    save_video: bool = True,
    device: Optional[str] = None
):
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    
    # Load checkpoint
    completed_sequences = []
    sequence_stats = []
    aggregated_metrics = {}

    if resume and os.path.exists(checkpoint_path):
        print(f"[BATCH] Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
            completed_sequences = ckpt.get("completed_sequences", [])
            sequence_stats = ckpt.get("sequence_stats", [])
            aggregated_metrics = ckpt.get("aggregated_metrics", {})
    else:
        print("[BATCH] Starting fresh batch evaluation.")

    # List sequences in dataset_root
    # Assuming dataset_root contains subfolders, each being a sequence (like VisDrone2019-MOT-val/sequences)
    # If the root itself is a sequence, we handle that as a list of 1.
    all_sequences = []
    if os.path.isfile(os.path.join(dataset_root, "gt.txt")) or any(f.endswith('.jpg') for f in os.listdir(dataset_root)):
        all_sequences = [dataset_root]
    else:
        for entry in sorted(os.listdir(dataset_root)):
            full_path = os.path.join(dataset_root, entry)
            if os.path.isdir(full_path):
                all_sequences.append(full_path)

    print(f"[BATCH] Found {len(all_sequences)} sequences to evaluate.")

    count_since_last_report = 0

    for seq_path in all_sequences:
        seq_name = os.path.basename(os.path.normpath(seq_path))
        if seq_name in completed_sequences:
            print(f"[BATCH] Skipping completed sequence: {seq_name}")
            continue

        # Auto-detect annotations in the standard VisDrone structure
        # (e.g. VisDrone2019-MOT-val/annotations/seq_name.txt)
        gt_path = None
        standard_gt = os.path.join(os.path.dirname(dataset_root), "annotations", f"{seq_name}.txt")
        if os.path.isfile(standard_gt):
            gt_path = standard_gt

        seq_out_dir = os.path.join(output_dir, seq_name)
        print(f"\n[BATCH] ---> Processing Sequence: {seq_name}")
        if gt_path:
            print(f"[BATCH] Found GT: {gt_path}")
        else:
            print(f"[BATCH] Warning: No GT found for {seq_name}. Tracking metrics will be 0.")

        try:
            results_json = evaluate_sequence(
                sequence_dir=seq_path,
                output_dir=seq_out_dir,
                gt_path=gt_path,
                save_video=save_video,
                planner_hz=planner_hz,
                max_frames=max_frames_per_seq,
                display=False,
                device=device
            )

            # Load sequence results
            with open(results_json, "r", encoding="utf-8") as f:
                res = json.load(f)
                
            stat = {
                "seq_name": seq_name,
                "frames": res.get("summary", {}).get("total_frames", 0),
                "duration_s": res.get("summary", {}).get("total_duration_s", 0),
                "perc_lat_ms": res.get("perception", {}).get("mean_latency_ms", 0),
                "miou": res.get("tracking", {}).get("mean_iou", 0),
                "mostly_tracked": res.get("tracking", {}).get("mostly_tracked_ratio", 0) * 100,
                "mostly_lost": res.get("tracking", {}).get("mostly_lost_ratio", 0) * 100
            }
            sequence_stats.append(stat)
            completed_sequences.append(seq_name)
            
            # Update aggregated
            aggregated_metrics = update_aggregated_metrics(aggregated_metrics, sequence_stats)
            
            # Save Checkpoint
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({
                    "completed_sequences": completed_sequences,
                    "sequence_stats": sequence_stats,
                    "aggregated_metrics": aggregated_metrics
                }, f, indent=2)
            
            count_since_last_report += 1

            # Generate report every 50 sequences
            if count_since_last_report >= 50:
                report_path = os.path.join(output_dir, f"batch_report_partial_{len(completed_sequences)}.pdf")
                generate_pdf_report(report_path, aggregated_metrics, sequence_stats)
                count_since_last_report = 0

        except Exception as e:
            print(f"[BATCH] Error processing sequence {seq_name}: {e}")
            # Do not add to completed_sequences, so it can be retried on --resume

    # Final Report
    if len(sequence_stats) > 0:
        final_report_path = os.path.join(output_dir, "batch_report_final.pdf")
        generate_pdf_report(final_report_path, aggregated_metrics, sequence_stats)
        print(f"\n[BATCH] All processing complete. Final report generated at {final_report_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluator with Checkpointing and PDF Reports")
    parser.add_argument("--dataset-root", "-d", type=str, required=True, help="Root folder containing sequence folders")
    parser.add_argument("--output-dir", "-o", type=str, default="batch_eval_output", help="Output directory for checkpoints and reports")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.json in output-dir")
    parser.add_argument("--planner-hz", type=float, default=2.5, help="Planner frequency")
    parser.add_argument("--max-frames-per-seq", type=int, default=None, help="Max frames to process per sequence")
    parser.add_argument("--no-video", dest="save_video", action="store_false", help="Disable video recording")
    parser.add_argument("--device", type=str, default=None, help="Hardware compute device ('cuda' or 'cpu')")
    parser.set_defaults(save_video=True)
    
    args = parser.parse_args()
    
    batch_evaluate(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        resume=args.resume,
        planner_hz=args.planner_hz,
        max_frames_per_seq=args.max_frames_per_seq,
        save_video=args.save_video,
        device=args.device
    )

if __name__ == "__main__":
    main()
