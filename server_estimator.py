#!/usr/bin/env python3
import zmq
import json
import time
import numpy as np

from ekf import EKFEstimator

from topics import TOPIC_COMMANDS, TOPIC_ESTIMATION, TOPIC_TELEMETRY

def main():
    context = zmq.Context()

    # Connect to PyBullet Simulation Server
    socket_sub = context.socket(zmq.SUB)
    socket_sub.setsockopt(zmq.CONFLATE, 1) 
    socket_sub.connect("tcp://localhost:5555")
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, TOPIC_TELEMETRY)
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, TOPIC_COMMANDS) 


    socket_pub = context.socket(zmq.PUB)
    socket_pub.connect("tcp://localhost:5556")

    # Initialize Estimator and Memory Variables
    ekf = EKFEstimator()
    prev_ang_vel = np.zeros(3)
    
    # Define rigid frame inertia (From your PyBullet URDF)
    inertia_tensor = (0.5, 0.5, 0.8) # Ixx, Iyy, Izz

    print("[ESTIMATOR] Server running. Listening for telemetry...")

    # State Holders
    latest_telem = None
    latest_wrench = None
    prev_ang_vel = np.zeros(3)
    inertia_tensor = (0.5, 0.5, 0.8)

    while True:
        try:
            # 1. Receive the next message in the queue
            raw_message = socket_sub.recv_string()
            topic, json_data = raw_message.split(" ", 1)
            
            # 2. Route the data based on the topic
            if topic == "telemetry":
                latest_telem = json.loads(json_data)
            elif topic == "wrench":
                latest_wrench = json.loads(json_data)

            # 3. Only run the EKF if we have fresh data for BOTH
            if latest_telem is not None and latest_wrench is not None:
                
                # Extract Telemetry
                dt = latest_telem["deltaTime"]
                current_ang_vel = np.array([
                    latest_telem["frame"]["angularVelocity"]["x"],
                    latest_telem["frame"]["angularVelocity"]["y"],
                    latest_telem["frame"]["angularVelocity"]["z"]
                ])

                # Extract Wrench
                F_total = latest_wrench["F_total"] 
                tau_geom = np.array([latest_wrench["tau_x"], latest_wrench["tau_y"]])

                # Calculate Angular Acceleration
                if dt > 0:
                    angular_accel = (current_ang_vel - prev_ang_vel) / dt
                else:
                    angular_accel = np.zeros(3)
                
                prev_ang_vel = current_ang_vel

                # Run EKF Update
                if F_total > 5.0: 
                    estimated_cog = ekf.update(
                        F_total=F_total,
                        tau_geom=tau_geom,
                        omega=current_ang_vel,
                        angular_accel=angular_accel,
                        inertia=inertia_tensor
                    )

                    print(f"Estimated Offset {estimated_cog[0]}{estimated_cog[1]}")
                else:
                    estimated_cog = ekf.x.flatten()

                # Send Estimate back to Port 5556
                estimate_payload = {
                    "time": latest_telem["time"],
                    "estimated_cog_x": float(estimated_cog[0]),
                    "estimated_cog_y": float(estimated_cog[1])
                }
                socket_pub.send_string(f"{TOPIC_ESTIMATION} {json.dumps(estimate_payload)}")

                # 4. Clear the state holders to wait for the next paired physics step
                latest_telem = None
                latest_wrench = None

        except KeyboardInterrupt:
            print("\nShutting down")
        finally:
            socket_pub.close()
            socket_sub.close()
            context.term()

if __name__ == "__main__":
    main()