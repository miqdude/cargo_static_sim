#!/usr/bin/env python3
import zmq
import json
import time
import numpy as np

from mellinger_control import MellingerControl
from geometric_control import GeometricControl

from waypoint import WaypointManager
from utils import get_trajectory, calculate_payload_offset

from topics import TOPIC_COMMANDS, TOPIC_TELEMETRY, TOPIC_ESTIMATION

# --- Configuration ---
FRAME_MASS = 6.8
PAYLOAD_MASS = 1.0  # Assumed static payload since PINN is removed
TOTAL_MASS = FRAME_MASS + PAYLOAD_MASS
RATE_HZ = 240.0
TIME_STEP = 1.0 / RATE_HZ

def main():
    context = zmq.Context()

    # 1. Listen to PyBullet Telemetry
    socket_sub = context.socket(zmq.SUB)
    socket_sub.setsockopt(zmq.CONFLATE, 1) 
    socket_sub.connect("tcp://localhost:5555") 
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, TOPIC_TELEMETRY)
    
    # 2. Listen to Estimator Updates
    socket_est = context.socket(zmq.SUB)
    socket_est.setsockopt(zmq.CONFLATE, 1) 
    socket_est.connect("tcp://localhost:5557") # Connects to the Estimator
    socket_est.setsockopt_string(zmq.SUBSCRIBE, TOPIC_ESTIMATION)

    # 3. Publish Thrust Commands
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind("tcp://*:5556")  # MUST BE BIND! So Simulator and Estimator can both connect to it.

    poller = zmq.Poller()
    poller.register(socket_sub, zmq.POLLIN)
    poller.register(socket_est, zmq.POLLIN)
    
    # Define standard Z-Up waypoints (e.g., hover at 7 meters)
    mission_waypoints = [
        [0.0, 0.0, 7.0],  
    ]

    # # 1. Initial Takeoff Waypoint
    # mission_waypoints = [
    #     [0.0, 0.0, 7.0], 
    # ]

    # # 2. Dynamically Generate Figure 8 Waypoints
    # num_points = 30
    # scale = 5.0       # How wide the figure 8 is in meters
    # altitude = 7.0    # Flight height

    # for i in range(num_points):
    #     # t goes from 0 to 2*PI
    #     t = (i / num_points) * (2 * np.pi)
        
    #     # Parametric math for a Figure 8 pattern
    #     x = scale * np.sin(t)
    #     y = scale * np.sin(t) * np.cos(t)
    #     z = altitude
        
    #     mission_waypoints.append([float(x), float(y), float(z)])

    # # 3. (Optional) Add a final waypoint to return to center and hover
    # mission_waypoints.append([0.0, 0.0, 7.0])

    mission_waypoints = [
        [0.0, 0.0, 5.0],   # Takeoff straight up 5 meters
        [5.0, 0.0, 5.0],   # Move 5 meters forward
        [5.0, 5.0, 5.0],   # Move 5 meters right
        [0.0, 5.0, 5.0],   # Move backwards
        [0.0, 0.0, 5.0]    # Return home
    ]

    # Initialize the controllers
    waypoint_manager = WaypointManager(mission_waypoints, acceptance_radius=0.3, max_speed=2.0)
    controller = MellingerControl(mass_total=TOTAL_MASS)
    # controller = GeometricControl(TOTAL_MASS)

    print("Controller Started...")
    print("(Press Ctrl+C to stop).")

    try:
        time_now = 0.0
        latest_telem = None
        est_offset_x = 0.0
        est_offset_y = 0.0
        est_mass = 0.0 # estimated payload mass

        while True:
            # Poll both sockets (timeout in milliseconds)
            socks = dict(poller.poll(timeout=10))

            if not socks:
                continue # Skip this loop iteration if PyBullet hasn't sent a new frame

            # Check if new telemetry arrived
            if socket_est in socks:
                raw_telem = socket_est.recv_string()
                _, json_telem = raw_telem.split(" ", 1)
                latest_est = json.loads(json_telem)
                raw_system_cog_x = latest_est["estimated_cog_x"]
                raw_system_cog_y = latest_est["estimated_cog_y"]

                # harusnya diconvert jadi offset dulu
                payload_mass_guess = max(0.05, TOTAL_MASS - FRAME_MASS)
                est_offset_x, est_offset_y, _ = calculate_payload_offset(FRAME_MASS, payload_mass_guess, [raw_system_cog_x, raw_system_cog_y, 0])

            if socket_sub in socks:
                raw_telem = socket_sub.recv_string()
                _, json_telem = raw_telem.split(" ", 1)
                latest_telem = json.loads(json_telem)

            if latest_telem is not None:
                time_delta = latest_telem["deltaTime"]
                composite_state = latest_telem["frame"]
                
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
                # using waypoints
                # target_pos, target_vel, target_acc, target_rpy, target_ang_vel = waypoint_manager.get_target_state(
                #     pos_np, time_delta
                # )

                # Ensure yaw is scalar
                # target_yaw_scalar = target_rpy[2] if isinstance(target_rpy, (list, np.ndarray)) else target_rpy

                # using trajectory generation
                target_pos, target_vel, target_acc, target_rpy, target_ang_vel = get_trajectory(time_now, "hover")

                # =========================================================
                # CONTROL ALLOCATION
                # =========================================================
                print(f"estimated offset {est_offset_x} {est_offset_y} estimated mass : {est_mass}")
                thrust_commands, torque_commands, wrench = controller.get_follower_commands(
                    [est_offset_x, est_offset_y], # Hardcoded offset guess (assuming perfect CoG)
                    time_delta,
                    pos_np,
                    rot_np,
                    vel_np,
                    ang_vel_np,
                    target_pos,
                    target_vel,
                    target_ang_vel,
                    target_acc,
                    target_rpy,
                    FRAME_MASS,
                    max(0.0, est_mass + FRAME_MASS), # payload mass
                )
                
                response = {"thrusts": thrust_commands, "torques": torque_commands, "wrench": wrench}
                json_response = json.dumps(response)

                # Send back to PyBullet
                socket_pub.send_string(f"{TOPIC_COMMANDS} {json_response}")

                time_now += TIME_STEP

    except KeyboardInterrupt:
        print("\nShutting down gracefully.")
    finally:
        socket_pub.close()
        socket_sub.close()
        socket_est.close()
        context.term()

if __name__ == "__main__":
    main()