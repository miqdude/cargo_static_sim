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

from topics import TOPIC_COMMANDS, TOPIC_TELEMETRY, TOPIC_ESTIMATION

# --- PlotJuggler UDP Setup ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PLOTJUGGLER_ADDRESS = ('127.0.0.1', 9870)

# --- Constants ---
RATE_HZ = 240.0
TIME_STEP = 1.0 / RATE_HZ
GRAVITY_MSS = 9.81
SIMULATION_DURATION = 120 # simulation time in seconds
TOTAL_EPISODES = 1
MAX_DRONE_THRUST = 50
TOTAL_SYSTEM_MASS = 7.8
FRAME_MASS = 6.8
PAYLOAD_MASS = 1.0
WIND_DISTURBANCE = True
ENABLE_PAYLOAD = False

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

log_id = None

# --- PyBullet initialization ---
if args.vis:
    p.connect(p.GUI) # showing visualization
    # PyBullet Recording to video requires GUI
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "simulation_recording.mp4")
else:
    p.connect(p.DIRECT)

p.setTimeStep(TIME_STEP)
p.setGravity(0, 0, -GRAVITY_MSS)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")


# Initialize the Controller
leader = MellingerControl(mass_total=TOTAL_SYSTEM_MASS)

env = PayloadEnv(args.f, args.fl, leader, p, ENABLE_PAYLOAD,
                GRAVITY_MSS, debug=args.debug, drone_max_thrust=MAX_DRONE_THRUST)

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
socket_sub.connect("tcp://localhost:5556")
socket_sub.setsockopt_string(zmq.SUBSCRIBE, TOPIC_COMMANDS)

# 2. Listen to Estimator Updates
socket_est = context.socket(zmq.SUB)
socket_est.setsockopt(zmq.CONFLATE, 1) 
socket_est.connect("tcp://localhost:5557") # Connects to the Estimator
socket_est.setsockopt_string(zmq.SUBSCRIBE, TOPIC_ESTIMATION)

poller = zmq.Poller()
poller.register(socket_sub, zmq.POLLIN)
poller.register(socket_est, zmq.POLLIN)

print("="*50)
print("PyBullet ZMQ Environment Server Running...")
print("Publishing state on tcp://*:5555")
print("Listening for control commands on tcp://*:5556")
print("Press Ctrl+C to stop.")

r_arms = {
    'iris_1': np.array([ 0.5,  0.5, 0.05]), 
    'iris_2': np.array([ 0.5, -0.5, 0.05]), 
    'iris_3': np.array([-0.5,  0.5, 0.05]), 
    'iris_4': np.array([-0.5, -0.5, 0.05])  
}

try:
    PATH_TRAJECTORY = "figure8"
    base_frame_inertia = [1.515, 1.515, 3.000] 

    curr_episode = 0
    now = datetime.now()
    filename_timestamp = now.strftime("%Y%m%d_%H%M%S")
    txt_log = ""

    # INTRA-EPISODE HISTORY (For Plotting and CSV)
    history_time = []
    history_true_mass = [] 
    ep_true_Jxx, ep_true_Jyy = [], []
    ep_true_offset_x, ep_true_offset_y = [], []

    # give default values at the beginning
    latest_est = {
        "estimated_mass": 6.8,
        "estimated_offset_x": 0.0,
        "estimated_offset_y": 0.0,
    }

    hist_est_mass = []
    hist_est_offset_x, hist_est_offset_y = [], []
    
    # 3D tracking lists
    ep_target_pos = []
    ep_actual_x, ep_actual_y, ep_actual_z = [], [], []
    ep_actual_rpy = []

    # ==========================================
    # THE EPISODIC LOOP
    # ==========================================
    while curr_episode < TOTAL_EPISODES:
        print("*"*36)
        print(f"{curr_episode}-th EPISODE IS STARTING")
        
        true_offset, true_mass = env.reset()
        print(f"True CoG Anomaly at: X={true_offset[0]:.3f}, Y={true_offset[1]:.3f}")

        time_now = 0.0
        step_counter = 0

        # Static assumption since AI estimation is removed
        guess_mass = FRAME_MASS
        guess_x, guess_y = 0.0, 0.0

        # initiate commands
        thrust_commands = {name: 0.0 for name in r_arms.keys()}
        torque_commands = {name: np.zeros(3) for name in r_arms.keys()}

        print(f"thrust commands {thrust_commands}")

        # ==========================================
        # THE CONTINUOUS FLIGHT LOOP
        # ==========================================
        while time_now <= SIMULATION_DURATION:

            # poll data
            socks = dict(poller.poll(timeout=1))

            if socket_sub in socks:
                raw_telem = socket_sub.recv_string()
                _, json_data = raw_telem.split(" ", 1)
                latest_commands = json.loads(json_data)
                thrust_commands = latest_commands["thrusts"]
                torque_commands = latest_commands["torques"]

                # tracking
                ep_target_pos.append(latest_commands["target_pos"])

                # calculated offset from controller
                latest_est["estimated_offset_x"] = latest_commands["calculated_offset_x"]
                latest_est["estimated_offset_y"] = latest_commands["calculated_offset_y"]

            if socket_est in socks:
                raw_telem = socket_est.recv_string()
                _, json_data = raw_telem.split(" ", 1)
                data_ = json.loads(json_data)

                # Update the guessed mass since it will affect the dynamics
                latest_est["estimated_mass"] = data_["estimated_mass"]
                
            pos, current_rpy, current_vel, current_ang_vel = env.step(
                thrust_commands,
                torque_commands,
                WIND_DISTURBANCE,
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
                    "offset": [true_offset[0], true_offset[1]],
                    "frameMass": FRAME_MASS,
                    "payloadMass": guess_mass - FRAME_MASS
                }
            }
            json_data = json.dumps(state_payload)

            socket_pub.send_string(f"{TOPIC_TELEMETRY} {json_data}")

            # ==========================================
            # PLOTTING
            # ==========================================
            hist_est_mass.append(latest_est["estimated_mass"])
            hist_est_offset_x.append(latest_est["estimated_offset_x"])
            hist_est_offset_y.append(latest_est["estimated_offset_y"])

            history_true_mass.append(TOTAL_SYSTEM_MASS)

            ep_actual_x.append(pos[0])
            ep_actual_y.append(pos[1])
            ep_actual_z.append(pos[2])
            ep_actual_rpy.append(current_rpy)

            # calculate the true Inertia
            true_j_xx, true_j_yy, true_j_zz = calculate_true_principal_inertia(base_frame_inertia, true_mass, [true_offset[0], true_offset[1], 0])
            ep_true_Jxx.append(true_j_xx)
            ep_true_Jyy.append(true_j_yy)

            # =========================================================
            # RECORD FRAME-BY-FRAME HISTORY
            # =========================================================
            true_tot_mass = FRAME_MASS + true_mass 

            history_time.append(time_now)
            
            ep_true_offset_x.append(true_offset[0])
            ep_true_offset_y.append(true_offset[1])

            time_now += TIME_STEP
            step_counter += 1

            # Add sleep otherwise the simulation will be finished in an instant
            if args.vis:
                time.sleep(TIME_STEP)

        curr_episode += 1

    print("\n--- Full Experimental Suite Completed ---")

# Graceful Exit
except KeyboardInterrupt:
    print("\n--- Simulation stopped by user ---")
finally:
    # Stop logging and save the video
    if args.vis:
        p.stopStateLogging(log_id)

    # Disconnect PyBullet
    p.disconnect()

    # ==========================================
    # RENDER FINAL STATIC PLOT FOR THE EPISODE
    # ==========================================
    # fix the dimension issue
    if len(history_time) > 0: 

        target_x = [p[0] for p in ep_target_pos]
        target_y = [p[1] for p in ep_target_pos]
        target_z = [p[2] for p in ep_target_pos]
        
        fig = plt.figure(figsize=(12, 15))
        gs = GridSpec(6, 1, figure=fig, hspace=0.4)
        plt.suptitle(f"Episode {curr_episode} Offset X: {true_offset[0]} Y: {true_offset[1]} Trajectory Tracking", fontsize=16)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[3, 0])
        ax5 = fig.add_subplot(gs[4, 0])
        ax6 = fig.add_subplot(gs[5, 0])

        # Subplot 1: Position over time
        ax1.plot(history_time, ep_actual_x, color="green",  label='Actual X', alpha=0.7)
        ax1.plot(history_time, ep_actual_y, color="blue",  label='Actual Y', linestyle="--", alpha=0.7)
        ax1.plot(history_time, ep_actual_z, color="purple",  label='Actual Z', linestyle=":", alpha=0.7)
        ax1.set(title='XYZ Position Over Time', xlabel='Time (s)', ylabel='Position (m)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Subplot 2: XY Plane
        ax2.plot(target_x, target_y, color="black",  label='Target Position', linewidth=3, linestyle="-")
        ax2.plot(ep_actual_x, ep_actual_y, color="red",  label='Actual Position', linewidth=2, linestyle="--")
        ax2.set(title='Top-Down View (XY Plane)', xlabel='X-axis (m)', ylabel='Y-axis (m)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Subplot 3: XZ Plane
        ax3.plot(target_x, target_z, color="black",  label='Target Position', linewidth=3, linestyle="-")
        ax3.plot(ep_actual_x, ep_actual_z, color="red",  label='Actual Position', linewidth=2, linestyle="--")
        ax3.set(title='Side View (XZ Plane)', xlabel='X-axis (m)', ylabel='Z-axis (m)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Mass converging over time
        ax4.plot(history_time, history_true_mass, color="black",  label='True Mass', linewidth=3, linestyle="-")
        ax4.plot(history_time, hist_est_mass, color="red",  label='Estimated Mass', linewidth=2, linestyle="--")
        ax4.set(title='Total System Mass Estimation Over Time', xlabel='Time (s)', ylabel='Mass (kg)')
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        # Offset converging over time
        ax5.plot(history_time, ep_true_offset_x, color="black",  label='True Offset', linewidth=3, linestyle="-")
        ax5.plot(history_time, hist_est_offset_x, color="red",  label='Estimated Offset', linewidth=2, linestyle="--")
        ax5.set(title='X Offset Estimation Over Time', xlabel='Time (s)', ylabel='meters (m)')
        ax5.set_ylim([-0.5, 2.0])
        ax5.grid(True, alpha=0.3)
        ax5.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        ax6.plot(history_time, ep_true_offset_y, color="black",  label='True Offset', linewidth=3, linestyle="-")
        ax6.plot(history_time, hist_est_offset_y, color="red",  label='Estimated Offset', linewidth=2, linestyle="--")
        ax6.set(title='Y Offset Estimation Over Time', xlabel='Time (s)', ylabel='meters (m)')
        ax6.set_ylim([-0.5, 2.0])
        ax6.grid(True, alpha=0.3)
        ax6.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))

        # plt.show()
        fig.savefig(f"./flight_data_{PATH_TRAJECTORY}.png", bbox_inches='tight')

        # ==========================================
        # EXPORT EPISODES TO CSV
        # ==========================================
        episode_data = {
            "time_sec": history_time,
            "true_mass_kg": history_true_mass,
            "true_offset_x_m": ep_true_offset_x,
            "true_offset_y_m": ep_true_offset_y,
            "actual_x_m": ep_actual_x,
            "actual_y_m": ep_actual_y,
            "actual_z_m": ep_actual_z,
        }

        df = pd.DataFrame(episode_data)
        csv_filename = f"./flight_data_{PATH_TRAJECTORY}.csv"
        df.iloc[::10, :].to_csv(csv_filename, index=False)
        print(f"Successfully saved {csv_filename}")