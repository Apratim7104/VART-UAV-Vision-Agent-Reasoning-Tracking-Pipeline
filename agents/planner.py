import json
import time
from typing import Any, Dict, List
import ollama


class PlannerAgent:
    """
    Local CPU-based LLM Planner Agent using Ollama and Qwen2.5-1.5B.
    Evaluates tracking JSON state trajectories and emits tactical flight directives.
    """

    SYSTEM_PROMPT = """
You are an autonomous UAV flight planner agent operating an aerial tracking drone.
Your job is to analyze time-series tracking data from an onboard vision system and select tactical actions.

AVAILABLE ACTIONS:
1. "SEARCH": Execute a search pattern (yaw sweep) because no valid targets are detected.
2. "CENTER": Target detected but offset (|center_offset_normalized| > 0.25). Adjust heading/position.
3. "TRACK": Target is reasonably centered (|center_offset_normalized| <= 0.25). Follow target trajectory.
4. "HOLD": Hover and wait for system stabilization.

DECISION RULES:
- If active_tracks is empty -> ACTION = "SEARCH"
- If target's center_offset_normalized magnitude for X or Y > 0.25 -> ACTION = "CENTER"
- If target's center_offset_normalized magnitude for X and Y <= 0.25 -> ACTION = "TRACK"
- Select the primary_target_id as the track ID of the target closest to frame center.

OUTPUT SCHEMA:
Respond strictly with raw JSON matching this format:
{
  "action": "SEARCH|CENTER|TRACK|HOLD",
  "primary_target_id": int or null,
  "reasoning": "short 1-sentence rationale",
  "recommended_yaw_rate_cmd": float between -1.0 and 1.0,
  "recommended_altitude_cmd": float between -1.0 and 1.0
}
"""

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        min_call_interval_s: float = 0.5,
    ):
        self.model_name = model_name
        self.min_call_interval_s = min_call_interval_s
        self.last_call_time = 0.0
        self.last_cached_plan: Dict[str, Any] = {
            "action": "HOLD",
            "primary_target_id": None,
            "reasoning": "Planner initializing...",
            "recommended_yaw_rate_cmd": 0.0,
            "recommended_altitude_cmd": 0.0,
        }

    def plan(
        self,
        active_tracks: List[Dict[str, Any]],
        force_update: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluates active tracks using local CPU LLM inference via Ollama.
        """
        now = time.time()
        elapsed = now - self.last_call_time

        # Throttle frequency if called faster than min_call_interval_s
        if not force_update and (elapsed < self.min_call_interval_s):
            return self.last_cached_plan

        # Pre-filter: Sort tracks by Euclidean distance from frame center and keep Top-3
        if len(active_tracks) > 3:
            active_tracks = sorted(
                active_tracks,
                key=lambda t: (
                    t.get("center_offset_normalized", [0.0, 0.0])[0] ** 2
                    + t.get("center_offset_normalized", [0.0, 0.0])[1] ** 2
                ),
            )[:3]

        payload = {
            "num_active_tracks": len(active_tracks),
            "active_tracks": active_tracks,
        }

        user_prompt = f"Current Trajectory Context:\n{json.dumps(payload)}"

        try:
            start_t = time.time()
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",  # Forces strict JSON structure output
                options={
                    "temperature": 0.1,
                    "num_ctx": 2048,
                },
            )

            latency_ms = (time.time() - start_t) * 1000.0
            plan_json = json.loads(response["message"]["content"])

            self.last_cached_plan = plan_json
            self.last_call_time = time.time()

            print(f"[PlannerAgent] CPU Inference completed in {latency_ms:.1f} ms")
            return plan_json

        except Exception as e:
            print(f"[PlannerAgent] Ollama error: {e}. Running fallback rule engine.")
            return self._rule_based_fallback(active_tracks)

    def _rule_based_fallback(self, active_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deterministic safety fallback engine during server or parsing issues."""
        if not active_tracks:
            return {
                "action": "SEARCH",
                "primary_target_id": None,
                "reasoning": "Fallback: No active tracks found.",
                "recommended_yaw_rate_cmd": 0.3,
                "recommended_altitude_cmd": 0.0,
            }

        # Select the target closest to center (Euclidean distance in normalized offsets)
        target = min(
            active_tracks,
            key=lambda t: (
                t.get("center_offset_normalized", [0.0, 0.0])[0] ** 2
                + t.get("center_offset_normalized", [0.0, 0.0])[1] ** 2
            ),
        )
        off_x, off_y = target.get("center_offset_normalized", [0.0, 0.0])
        action = "CENTER" if (abs(off_x) > 0.25 or abs(off_y) > 0.25) else "TRACK"

        return {
            "action": action,
            "primary_target_id": target["track_id"],
            "reasoning": f"Fallback Rule Engine active. Offset: [{off_x:.2f}, {off_y:.2f}].",
            "recommended_yaw_rate_cmd": round(off_x, 2),
            "recommended_altitude_cmd": round(-off_y, 2),
        }


if __name__ == "__main__":
    # Test bench script using simulated Kalman track data
    mock_tracks = [
        {
            "track_id": 101,
            "label": "car",
            "center_offset_normalized": [0.38, -0.05],
            "velocity_pixel_s": [12.0, 0.5],
            "age_frames": 10,
        }
    ]

    planner = PlannerAgent(min_call_interval_s=0.0)
    output_plan = planner.plan(mock_tracks, force_update=True)

    print("\n--- Local CPU Planner Test Output ---")
    print(json.dumps(output_plan, indent=2))