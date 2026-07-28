#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pybullet as p
import pybullet_data
import socket
import json
import pandas as pd
import time
import collections
import zmq
from datetime import datetime
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from mellinger_control import MellingerControl
from payload_env import PayloadEnv
from utils import get_trajectory, calculate_true_principal_inertia

# --- PlotJuggler UDP Setup ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PLOTJUGGLER_ADDRESS = ('127.0.0.1', 9870)

# --- Constants ---
RATE_HZ = 240.0
TIME_STEP = 1.0 / RATE_HZ
GRAVITY_MSS = 9.81
SIMULATION_DURATION = 20 # simulation time in seconds
TOTAL_EPISODES = 1000
MAX_DRONE_THRUST = 50
TOTAL_SYSTEM_MASS = 7.8
FRAME_MASS = 6.8

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
parser = argparse.ArgumentParser(description="Simulation of Quadrotors lifting a payload")

parser.add_argument("-f", type=str, default="final_assembly.urdf", help="file path to the environment")
parser.add_argument("-fl", type=str, default="virtual_leader.urdf", help="urdf file to the leader")
parser.add_argument("-r", type=str, help="flight data in csv")
parser.add_argument("--debug", type=bool, default=False, help="use visual debug in simulation")
parser.add_argument("--vis", type=bool, default=True, help="show simulation")

args = parser.parse_args()

# --- PyBullet initialization ---
if args.vis:
    p.connect(p.GUI) # showing visualization
else:
    p.connect(p.DIRECT)

p.setTimeStep(TIME_STEP)
p.setGravity(0, 0, -GRAVITY_MSS)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# Initialize the Controller
leader = MellingerControl(mass_total=TOTAL_SYSTEM_MASS)

env = PayloadEnv(args.f, args.fl, leader, p, GRAVITY_MSS, debug=args.debug, drone_max_thrust=MAX_DRONE_THRUST)
true_cog = env.reset()

print("="*50)
print(f"Simulation Environment running, duration {SIMULATION_DURATION} secs.")
print("Streaming to PlotJuggler on port 9870...")
print("Press Ctrl+C in the terminal to stop")

if args.r:
    print("RECORDING DATA to CSV")

flight_data = []

# ==========================================
# ZMQ SOCKET SETUP
# ==========================================
context = zmq.Context()

# Socket to publish PyBullet state to external controllers (Port 5555)
socket_pub = context.socket(zmq.PUB)
socket_pub.bind("tcp://*:5555")

# Socket to receive motor commands/wrenches from external controllers (Port 5556)
socket_sub = context.socket(zmq.SUB)
socket_sub.setsockopt(zmq.CONFLATE, 1)  # Always keep only the newest packet
socket_sub.bind("tcp://*:5556")
socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")

print("="*50)
print("PyBullet ZMQ Environment Server Running...")
print("Publishing state on tcp://*:5555")
print("Listening for control commands on tcp://*:5556")
print("Press Ctrl+C to stop.")

try:
    PATH_TRAJECTORY = "figure_eight_3d"
    base_frame_inertia = [1.515, 1.515, 3.000] 

    curr_episode = 0
    now = datetime.now()
    filename_timestamp = now.strftime("%Y%m%d_%H%M%S")
    txt_log = ""

    # ==========================================
    # THE EPISODIC LOOP
    # ==========================================
    while curr_episode < TOTAL_EPISODES:
        print("*"*36)
        print(f"{curr_episode}-th EPISODE IS STARTING")
        
        true_cog, true_mass = env.reset()
        print(f"True CoG Anomaly at: X={true_cog[0]:.3f}, Y={true_cog[1]:.3f}")

        time_now = 0.0
        step_counter = 0

        # INTRA-EPISODE HISTORY (For Plotting and CSV)
        history_time = []
        history_true_mass = [] 
        ep_true_Jxx, ep_true_Jyy = [], []
        ep_true_offset_x, ep_true_offset_y = [], []
        
        # 3D tracking lists
        ep_actual_x, ep_actual_y, ep_actual_z = [], [], []
        ep_target_x, ep_target_y, ep_target_z = [], [], []
        ep_actual_rpy = []

        # Static assumption since AI estimation is removed
        guess_mass = FRAME_MASS
        guess_x, guess_y = 0.0, 0.0

        # ==========================================
        # THE CONTINUOUS FLIGHT LOOP
        # ==========================================
        while time_now <= SIMULATION_DURATION:
            
            target_pos, target_vel, target_acc, target_rpy, target_ang_vel = get_trajectory(time_now, path_type=PATH_TRAJECTORY)
            
            # Extract scalar yaw
            target_yaw_scalar = target_rpy[2]
            
            pos, current_rpy, current_vel, current_ang_vel, current_acc, current_ang_acc, acting_force, follower_thrust_cmds, follower_torque_cmds = env.step(
                FRAME_MASS, max(0.0, guess_mass - FRAME_MASS), TIME_STEP, time_now, 
                target_pos, target_vel, target_ang_vel, target_acc, target_yaw_scalar, [guess_x, guess_y]
            )

            # 2. PACKAGE AND PUBLISH STATE OVER ZMQ (PUB)
            state_payload = {
                "time": time_now,
                "deltaTime": TIME_STEP,
                "frame": {
                    "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                    "velocity": {"x": current_vel[0], "y": current_vel[1], "z": current_vel[2]},
                    "rotation": {"x": current_rpy[0], "y": current_rpy[1], "z": current_rpy[2]},
                    "angularVelocity": {"x": current_ang_vel[0], "y": current_ang_vel[1], "z": current_ang_vel[2]}
                },
                "payloadState": {
                    "offset": [true_cog[0], true_cog[1]],
                    "frameMass": FRAME_MASS
                }
            }
            socket_pub.send_string(json.dumps(state_payload))

            # 3. NON-BLOCKING RECEIVE COMMANDS FROM EXTERNAL PROCESS (SUB)
            try:
                message = socket_sub.recv_string(flags=zmq.NOBLOCK)
                command_data = json.loads(message)
                if "thrusts" in command_data:
                    external_thrusts = command_data["thrusts"]
            except zmq.Again:
                # If no packet received this tick, reuse last command or handle fallback
                pass

            ep_actual_x.append(pos[0])
            ep_actual_y.append(pos[1])
            ep_actual_z.append(pos[2])
            ep_target_x.append(target_pos[0])
            ep_target_y.append(target_pos[1])
            ep_target_z.append(target_pos[2])
            ep_actual_rpy.append(current_rpy)

            # calculate the true Inertia
            true_j_xx, true_j_yy, true_j_zz = calculate_true_principal_inertia(base_frame_inertia, true_mass, [true_cog[0], true_cog[1], 0])
            ep_true_Jxx.append(true_j_xx)
            ep_true_Jyy.append(true_j_yy)

            # =========================================================
            # RECORD FRAME-BY-FRAME HISTORY
            # =========================================================
            true_tot_mass = FRAME_MASS + true_mass 

            history_time.append(time_now)
            history_true_mass.append(true_tot_mass)
            
            ep_true_offset_x.append(true_cog[0])
            ep_true_offset_y.append(true_cog[1])

            # Crash Detection
            distance_err = np.linalg.norm(np.array(pos) - np.array(target_pos))
            if abs(current_rpy[0]) > 1.5 or abs(current_rpy[1]) > 1.5 or distance_err > 5.0:
                print(f"Flight unstable at t={time_now:.2f}s. Ending rollout early...")
                break 

            time_now += TIME_STEP
            step_counter += 1

            # Add sleep otherwise the simulation will be finished in an instant
            if args.vis:
                time.sleep(TIME_STEP)

        print(f"Episode {curr_episode} Completed | Assumed Mass: {guess_mass:.3f} kg | Assumed CoG: ({guess_x:.3f}, {guess_y:.3f})")

        # ==========================================
        # RENDER FINAL STATIC PLOT FOR THE EPISODE
        # ==========================================
        if len(history_time) > 0: 
            
            fig = plt.figure(figsize=(12, 10))
            gs = GridSpec(3, 1, figure=fig, hspace=0.4)
            plt.suptitle(f"Episode {curr_episode}: Trajectory Tracking", fontsize=16)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, 0])

            # Subplot 1: Position over time
            ax1.plot(history_time, ep_target_x, color="red",  label='Target X')
            ax1.plot(history_time, ep_target_y, color="gold",  label='Target Y', linestyle="--")
            ax1.plot(history_time, ep_target_z, color="black",  label='Target Z', linestyle=":")
            ax1.plot(history_time, ep_actual_x, color="green",  label='Actual X', alpha=0.7)
            ax1.plot(history_time, ep_actual_y, color="blue",  label='Actual Y', linestyle="--", alpha=0.7)
            ax1.plot(history_time, ep_actual_z, color="purple",  label='Actual Z', linestyle=":", alpha=0.7)
            ax1.set(title='XYZ Position Over Time', xlabel='Time (s)', ylabel='Position (m)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

            # Subplot 2: XY Plane
            ax2.plot(ep_target_x, ep_target_y,  color="black",  label='Target Path', linewidth=2)
            ax2.plot(ep_actual_x, ep_actual_y, color="red",  label='Actual Path', linewidth=2, linestyle="--")
            ax2.set(title='Top-Down View (XY Plane)', xlabel='X-axis (m)', ylabel='Y-axis (m)')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

            # Subplot 3: XZ Plane
            ax3.plot(ep_target_x, ep_target_z,  color="black",  label='Target Altitude', linewidth=2)
            ax3.plot(ep_actual_x, ep_actual_z, color="red",  label='Actual Altitude', linewidth=2, linestyle="--")
            ax3.set(title='Side View (XZ Plane)', xlabel='X-axis (m)', ylabel='Z-axis (m)')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

            fig.savefig(f"./flight_data_{PATH_TRAJECTORY}_ep_{curr_episode}.png", bbox_inches='tight')
            # plt.show()

            # ==========================================
            # CALCULATE TRAJECTORY RMSE
            # ==========================================
            arr_target = np.array([ep_target_x, ep_target_y, ep_target_z])
            arr_actual = np.array([ep_actual_x, ep_actual_y, ep_actual_z])
            squared_errors = (arr_target - arr_actual) ** 2
            sum_squared_errors = np.sum(squared_errors, axis=0)
            rmse_trajectory = np.sqrt(np.mean(sum_squared_errors))
            
            rmse_per_axis = np.sqrt(np.mean(squared_errors, axis=1))
            rmse_x = rmse_per_axis[0]
            rmse_y = rmse_per_axis[1]
            rmse_z = rmse_per_axis[2]
            print(f"Episode {curr_episode} Trajectory RMSE: {rmse_trajectory:.4f} meters")

            # ==========================================
            # EXPORT EPISODES TO CSV
            # ==========================================
            episode_data = {
                "time_sec": history_time,
                "true_mass_kg": history_true_mass,
                "true_offset_x_m": ep_true_offset_x,
                "true_offset_y_m": ep_true_offset_y,
                "target_x_m": ep_target_x,
                "target_y_m": ep_target_y,
                "target_z_m": ep_target_z,
                "actual_x_m": ep_actual_x,
                "actual_y_m": ep_actual_y,
                "actual_z_m": ep_actual_z,
            }

            df = pd.DataFrame(episode_data)
            csv_filename = f"./flight_data_{PATH_TRAJECTORY}_ep_{curr_episode}.csv"
            df.iloc[::10, :].to_csv(csv_filename, index=False)
            print(f"Successfully saved {csv_filename}")

        # ==========================================
        # Logging Text 
        # ==========================================
        txt_log += "# ========================================== \n"
        txt_log += f"# Episode {curr_episode} \n"
        txt_log += "# ========================================== \n"
        txt_log += f"Offset True: x:{true_cog[0]:.4f} y:{true_cog[1]:.4f} \n"
        txt_log += f"Trajectory Loss RMSE: {rmse_trajectory:.4f} meters \n"
        txt_log += f"Trajectory x-axis loss: {rmse_x:.4f} meters \n"
        txt_log += f"Trajectory y-axis loss: {rmse_y:.4f} meters \n"
        txt_log += f"Trajectory z-axis loss: {rmse_z:.4f} meters \n\n"

        curr_episode += 1

    # Save log file
    log_file_path = f"training_log_{PATH_TRAJECTORY}_{filename_timestamp}.txt"
    with open(log_file_path, "a") as file:
        file.write(txt_log + "\n")

    print("\n--- Full Experimental Suite Completed ---")

# Graceful Exit
except KeyboardInterrupt:
    print("\n--- Simulation stopped by user ---")
    
    if args.r and len(flight_data) > 0:
        df = pd.DataFrame(flight_data)
        df.to_csv(args.r, index=False)
        print(f"Successfully saved final dataset to {args.r}!")
    else:
        print("No final dataset saved.")

    p.disconnect()