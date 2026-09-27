import time
from typing import Dict, Any, Optional, Tuple


class ControllerAgent:
    """
    Translates high-level Planner actions and spatial offset vectors (dx, dy)
    into low-level flight control velocity setpoints [vx, vy, vz, yaw_rate].
    """

    def __init__(
        self,
        k_p_yaw: float = 1.2,      # Proportional gain for yaw rotation
        k_p_alt: float = 0.8,      # Proportional gain for vertical rate
        k_p_forward: float = 1.0,  # Proportional gain for forward velocity
        max_vel_mps: float = 2.0,   # Max velocity limit (m/s)
        max_yaw_rate_rads: float = 0.5  # Max yaw rate limit (rad/s)
    ):
        self.k_p_yaw = k_p_yaw
        self.k_p_alt = k_p_alt
        self.k_p_forward = k_p_forward
        self.max_vel_mps = max_vel_mps
        self.max_yaw_rate_rads = max_yaw_rate_rads

    def compute_setpoints(
        self,
        planner_output: Dict[str, Any],
        primary_target: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates 4-DOF velocity commands based on active planner action and target offsets.
        """
        raw_action = planner_output.get("action", "HOLD") if isinstance(planner_output, dict) else "HOLD"
        action = str(raw_action).upper().strip()
        
        # Default velocity setpoints (Hover/Hold)
        vx = 0.0        # Forward (+) / Backward (-) velocity (m/s)
        vy = 0.0        # Lateral Right (+) / Left (-) velocity (m/s)
        vz = 0.0        # Ascend (+) / Descend (-) velocity (m/s)
        yaw_rate = 0.0  # Yaw Right (+) / Left (-) rate (rad/s)

        if action == "SEARCH":
            # Execute a gentle scanning yaw rotation while maintaining altitude
            yaw_rate = 0.25 * self.max_yaw_rate_rads

        elif action in ["CENTER", "TRACK"] and primary_target is not None:
            offsets = primary_target.get("center_offset_normalized", [0.0, 0.0])
            off_x = offsets[0] if len(offsets) > 0 else 0.0
            off_y = offsets[1] if len(offsets) > 1 else 0.0
            
            # Closed-loop Proportional Control Mapping
            # Normalized off_x (-1.0 to 1.0) maps to Yaw command
            yaw_rate = self._clamp(off_x * self.k_p_yaw, -self.max_yaw_rate_rads, self.max_yaw_rate_rads)
            
            # Normalized off_y (-1.0 to 1.0) maps to Vertical (altitude) adjustment
            # Negative off_y means object is higher in frame -> Drone should ascend (+)
            vz = self._clamp(-off_y * self.k_p_alt, -self.max_vel_mps, self.max_vel_mps)

            if action == "TRACK":
                # Advance forward when target is locked and centered
                vels = primary_target.get("velocity_pixel_s", [0.0, 0.0])
                target_vel_x = vels[0] if len(vels) > 0 else 0.0
                vx = self._clamp(0.5 + (abs(target_vel_x) * 0.01), 0.0, self.max_vel_mps)

        elif action == "HOLD":
            # Velocity remains zero for all axes
            pass

        return {
            "timestamp": round(time.time(), 2),
            "executed_action": action,
            "setpoints": {
                "vx_m_s": round(vx, 3),
                "vy_m_s": round(vy, 3),
                "vz_m_s": round(vz, 3),
                "yaw_rate_rad_s": round(yaw_rate, 3)
            }
        }

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(value, max_val))


if __name__ == "__main__":
    # Test script for ControllerAgent
    mock_planner_output = {
        "action": "CENTER",
        "primary_target_id": 101,
        "reasoning": "Target offset requires yaw correction."
    }

    mock_target = {
        "track_id": 101,
        "center_offset_normalized": [0.35, -0.10],
        "velocity_pixel_s": [10.0, 0.0]
    }

    controller = ControllerAgent()
    commands = controller.compute_setpoints(mock_planner_output, mock_target)

    print("\n--- Control Agent Output Setpoints ---")
    print(commands)

