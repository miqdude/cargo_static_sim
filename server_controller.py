#!/usr/bin/env python3
import zmq
import json
import time
import numpy as np

from mellinger_control import MellingerControl
from waypoint import WaypointManager

# --- Configuration ---
FRAME_MASS = 6.8
PAYLOAD_MASS = 1.0  # Assumed static payload since PINN is removed
TOTAL_MASS = FRAME_MASS + PAYLOAD_MASS

def main():
    context = zmq.Context()

    # 1. Connect to the PyBullet Publisher (Receives State)
    socket_sub = context.socket(zmq.SUB)
    socket_sub.setsockopt(zmq.CONFLATE, 1) 
    socket_sub.connect("tcp://localhost:5555") # Note: 'connect' instead of 'bind'
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, "") 

    # 2. Connect to the PyBullet Subscriber (Sends Thrusts)
    socket_pub = context.socket(zmq.PUB)
    socket_pub.connect("tcp://localhost:5556")
    
    # Define standard Z-Up waypoints (e.g., hover at 7 meters)
    mission_waypoints = [
        [0.0, 0.0, 7.0],  
    ]

    # Initialize the controllers
    waypoint_manager = WaypointManager(mission_waypoints, acceptance_radius=0.3, max_speed=2.0)
    controller = MellingerControl(mass_total=TOTAL_MASS)

    print("Classical Mellinger ZMQ Controller Started...")
    print("Waiting for telemetry from PyBullet (Press Ctrl+C to stop).")

    try:
        while True:
            try:
                message = socket_sub.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                # No data received from PyBullet yet on this loop iteration.
                # Sleep for 1 millisecond to prevent this while loop from maxing out 100% of your CPU core.
                time.sleep(0.001)
                continue
            
            state_data = json.loads(message)
            time_delta = state_data["deltaTime"]
            composite_state = state_data["frame"]
            
            # Extract basic dictionaries
            pos = composite_state["position"]
            vel = composite_state["velocity"]
            rot = composite_state["rotation"]
            ang_vel = composite_state["angularVelocity"]

            # Standard 1-to-1 Z-Up Mapping (No Unity Axis Swapping Required!)
            pos_np = np.array([pos['x'], pos['y'], pos['z']])
            vel_np = np.array([vel['x'], vel['y'], vel['z']])
            rot_np = np.array([rot['x'], rot['y'], rot['z']])
            ang_vel_np = np.array([ang_vel['x'], ang_vel['y'], ang_vel['z']])

            # =========================================================
            # TRAJECTORY GENERATION
            # =========================================================
            target_pos, target_vel, target_acc, target_rpy, target_ang_vel = waypoint_manager.get_target_state(
                pos_np, time_delta
            )

            # Ensure yaw is scalar
            target_yaw_scalar = target_rpy[2] if isinstance(target_rpy, (list, np.ndarray)) else target_rpy

            # =========================================================
            # CONTROL ALLOCATION
            # =========================================================
            thrust_commands, torque_commands, wrench = controller.get_follower_commands(
                [0.0, 0.0], # Hardcoded offset guess (assuming perfect CoG)
                time_delta,
                pos_np,
                rot_np,
                vel_np,
                ang_vel_np,
                target_pos,
                target_vel,
                target_ang_vel,
                target_acc,
                target_yaw_scalar,
                FRAME_MASS,
                PAYLOAD_MASS,
            )
            
            response = {"thrusts": thrust_commands, "torques": torque_commands}

            # Send back to PyBullet
            socket_pub.send_string(json.dumps(response))

    except KeyboardInterrupt:
        print("\nShutting down gracefully.")
    finally:
        socket_pub.close()
        socket_sub.close()
        context.term()

if __name__ == "__main__":
    main()