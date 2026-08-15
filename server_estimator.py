#!/usr/bin/env python3
import zmq
import json
import time
import numpy as np

from ekf import EKFEstimator
from rls import RLSEstimator

from topics import TOPIC_COMMANDS, TOPIC_ESTIMATION, TOPIC_TELEMETRY

def main():
    context = zmq.Context()

    # 1. DOWNLINK: Listen to PyBullet Telemetry
    sub_telem = context.socket(zmq.SUB)
    sub_telem.setsockopt(zmq.CONFLATE, 1) 
    sub_telem.connect("tcp://localhost:5555") 
    sub_telem.setsockopt_string(zmq.SUBSCRIBE, TOPIC_TELEMETRY)

    # 2. COMMANDS: Listen to Controller Wrenches
    sub_cmd = context.socket(zmq.SUB)
    sub_cmd.setsockopt(zmq.CONFLATE, 1)
    sub_cmd.connect("tcp://localhost:5556")
    sub_cmd.setsockopt_string(zmq.SUBSCRIBE, TOPIC_COMMANDS)

    # 3. UPLINK: Broadcast Estimates
    pub_est = context.socket(zmq.PUB)
    pub_est.bind("tcp://*:5557") # BIND because it acts as the host for estimates

    # Register the Poller
    poller = zmq.Poller()
    poller.register(sub_telem, zmq.POLLIN)
    poller.register(sub_cmd, zmq.POLLIN)

    # Initialize Estimator and Memory Variables
    ekf = EKFEstimator()
    rls = RLSEstimator()
    prev_ang_vel = np.zeros(3)
    
    # Define rigid frame inertia (From your PyBullet URDF)
    inertia_tensor = (0.5, 0.5, 0.8) # Ixx, Iyy, Izz

    print("[ESTIMATOR] Server running. Listening for telemetry...")

    # State Holders
    latest_telem = None
    latest_wrench = None
    prev_ang_vel = np.zeros(3)
    inertia_tensor = (0.5, 0.5, 0.8)

    try:
        while True:
            # Poll both sockets (timeout in milliseconds)
            socks = dict(poller.poll(timeout=10))

            # Check if new telemetry arrived
            if sub_telem in socks:
                raw_telem = sub_telem.recv_string()
                _, json_telem = raw_telem.split(" ", 1)
                latest_telem = json.loads(json_telem)

            # Check if new commands arrived
            if sub_cmd in socks:
                raw_cmd = sub_cmd.recv_string()
                _, json_cmd = raw_cmd.split(" ", 1)
                latest_wrench = json.loads(json_cmd)

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
                F_total = latest_wrench["wrench"][0] 
                tau_geom = np.array([latest_wrench["wrench"][1], latest_wrench["wrench"][2]])

                # Calculate Angular Acceleration
                if dt > 0:
                    angular_accel = (current_ang_vel - prev_ang_vel) / dt
                else:
                    angular_accel = np.zeros(3)
                
                prev_ang_vel = current_ang_vel

                # Run EKF Update
                if F_total > 5.0: 
                    # estimated_cog = ekf.update(
                    #     F_total=F_total,
                    #     tau_geom=tau_geom,
                    #     omega=current_ang_vel,
                    #     angular_accel=angular_accel,
                    #     inertia=inertia_tensor
                    # )
                    estimated_cog = rls.update(
                        F_total=F_total,
                        tau_geom=tau_geom,
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
                pub_est.send_string(f"{TOPIC_ESTIMATION} {json.dumps(estimate_payload)}")

                # 4. Clear the state holders to wait for the next paired physics step
                latest_telem = None
                latest_wrench = None

    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        sub_cmd.close()
        sub_telem.close()
        pub_est.close()
        context.term()

if __name__ == "__main__":
    main()