#!/usr/bin/env python3
import zmq
import json
import time
import torch
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import YOUR exact PINN code here
from pinn_inverse_estimator import PINNInverseEstimator, train_pinn_step
from topics import TOPIC_COMMANDS, TOPIC_ESTIMATION, TOPIC_TELEMETRY

WINDOW_SIZE = 50

def main():
    context = zmq.Context()

    # DOWNLINK: Listen to PyBullet Telemetry
    sub_telem = context.socket(zmq.SUB)
    sub_telem.setsockopt(zmq.CONFLATE, 1) 
    sub_telem.connect("tcp://localhost:5555") 
    sub_telem.setsockopt_string(zmq.SUBSCRIBE, TOPIC_TELEMETRY)

    # COMMANDS: Listen to Controller Wrenches
    sub_cmd = context.socket(zmq.SUB)
    sub_cmd.setsockopt(zmq.CONFLATE, 1)
    sub_cmd.connect("tcp://localhost:5556")
    sub_cmd.setsockopt_string(zmq.SUBSCRIBE, TOPIC_COMMANDS)

    # UPLINK: Broadcast Estimates
    pub_est = context.socket(zmq.PUB)
    pub_est.bind("tcp://*:5557")

    poller = zmq.Poller()
    poller.register(sub_telem, zmq.POLLIN)
    poller.register(sub_cmd, zmq.POLLIN)

    # Initialize Your PyTorch Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_model = PINNInverseEstimator().to(device)
    optimizer = optim.Adam(pinn_model.parameters(), lr=1e-3)
    
    history_buffer = deque(maxlen=WINDOW_SIZE)

    print(f"[PINN SERVER] Running on {device}. Waiting for {WINDOW_SIZE} frames...")

    latest_telem = None
    latest_wrench = None

    t_val = 0
    est_mass = 6.0
    est_cog_x = 0.0
    est_cog_y = 0.0
    est_j_xx = 0.0
    est_j_yy = 0.0
    est_j_zz = 0.0

    # --- FIX 1: Safely warm-start parameters WITHOUT breaking the optimizer or device ---
    with torch.no_grad():
        pinn_model.mass.data = torch.tensor([est_mass], device=device)
        pinn_model.cog_x.data = torch.tensor([est_cog_x], device=device)
        pinn_model.cog_y.data = torch.tensor([est_cog_y], device=device)

    try:
        hist_time, hist_pos, hist_vel = [], [], []
        hist_total_loss = []

        while True:
            socks = dict(poller.poll(timeout=10))

            if sub_telem in socks:
                raw_telem = sub_telem.recv_string()
                _, json_telem = raw_telem.split(" ", 1)
                latest_telem = json.loads(json_telem)

            if sub_cmd in socks:
                raw_cmd = sub_cmd.recv_string()
                _, json_cmd = raw_cmd.split(" ", 1)
                latest_wrench = json.loads(json_cmd)

            if latest_telem is not None and latest_wrench is not None:
                
                # Extract Data
                t_val = latest_telem["time"]
                f = latest_telem["frame"]
                pos, vel, rot, ang = f["position"], f["velocity"], f["rotation"], f["angularVelocity"]

                # print(f"pos {pos} vel {vel} rot {rot} ang {ang}")

                hist_time.append(t_val)
                hist_pos.append(pos)
                hist_vel.append(vel)
                
                state_12dof = [
                    pos["x"], pos["y"], pos["z"],
                    vel["x"], vel["y"], vel["z"],
                    rot["x"], rot["y"], rot["z"],
                    ang["x"], ang["y"], ang["z"]
                ]

                # Extract Wrench [F_total, tau_x, tau_y, tau_z]
                F_total = latest_wrench["wrench"][0]
                tau_x = latest_wrench["wrench"][1]
                tau_y = latest_wrench["wrench"][2]
                tau_z = latest_wrench["wrench"][3] if len(latest_wrench["wrench"]) > 3 else 0.0

                # Append to sliding window
                history_buffer.append({
                    "time": t_val,
                    "state": state_12dof,
                    "thrust": F_total,
                    "torques": [tau_x, tau_y, tau_z]
                })

                # Execute Your Training Step
                if F_total > 5.0 and len(history_buffer) == WINDOW_SIZE: 
                    
                    # Convert Buffer to Tensors
                    t_batch = torch.tensor([b["time"] for b in history_buffer], dtype=torch.float32, device=device).view(-1, 1)
                    t_batch.requires_grad_(True) # CRITICAL for autograd
                    
                    s_batch = torch.tensor([b["state"] for b in history_buffer], dtype=torch.float32, device=device)
                    f_batch = torch.tensor([b["thrust"] for b in history_buffer], dtype=torch.float32, device=device).view(-1, 1)
                    tau_batch = torch.tensor([b["torques"] for b in history_buffer], dtype=torch.float32, device=device)
                    
                    # Call YOUR function
                    # loss_tot, loss_d, loss_t, loss_r = train_pinn_step(
                    #     pinn_model, optimizer, 
                    #     t_batch, s_batch, 
                    #     f_batch, tau_batch
                    # )
                    for grad_step in range(5):
                        optimizer.zero_grad()
                        total_loss, data_loss, trans_loss, rot_loss = train_pinn_step(
                            pinn_model, optimizer, t_batch, s_batch, f_batch, tau_batch
                        )

                        print(f"PINN Loss: {total_loss.item():.4f}")

                        # update the history
                        hist_total_loss.append(total_loss.item())

                    with torch.no_grad():
                        # Drone is grounded or buffer is filling
                        est_cog_x = float(pinn_model.cog_x.item())
                        est_cog_y = float(pinn_model.cog_y.item())
                        
                        # Corrected to softplus to match the forward pass logic
                        est_mass = float(F.softplus(pinn_model.mass).item())

                        est_j_xx = float(F.softplus(pinn_model.J_xx).item())
                        est_j_yy = float(F.softplus(pinn_model.J_yy).item()) 
                        est_j_zz = float(F.softplus(pinn_model.J_zz).item())
                    
                    print(f"Estimated mass {est_mass})")

                # Estimations
                # EMA low pass filter to remove high frequency spikes
                ALPHA = 0.1
                est_mass = (1.0 - ALPHA) * est_mass + ALPHA * est_mass
                est_cog_x = (1.0 - ALPHA) * est_cog_x + ALPHA * est_cog_x
                est_cog_y = (1.0 - ALPHA) * est_cog_y + ALPHA * est_cog_y
                
                est_j_xx = (1.0 - ALPHA) * est_j_xx + ALPHA * est_j_xx
                est_j_yy = (1.0 - ALPHA) * est_j_yy + ALPHA * est_j_yy
                est_j_zz = (1.0 - ALPHA) * est_j_zz + ALPHA * est_j_zz

                estimations = {
                    "time": t_val,
                    "estimated_cog_x": est_cog_x,
                    "estimated_cog_y": est_cog_y,
                    "estimated_mass": est_mass,
                    "estimated_j_xx": est_j_xx,
                    "estimated_j_yy": est_j_yy,
                    "estimated_j_zz": est_j_zz,
                }

                # print(f"estimations {estimate_payload}")
                pub_est.send_string(f"{TOPIC_ESTIMATION} {json.dumps(estimations)}")

                # Clear for next tick
                latest_telem = None
                latest_wrench = None

    except KeyboardInterrupt:
        print("\nShutting down PINN Server")
    finally:
        sub_cmd.close()
        sub_telem.close()
        pub_est.close()
        context.term()

        # ==========================================
        # EXPORT EPISODES TO CSV
        # ==========================================
        episode_data = {
            "time": hist_time,
            "vel": hist_vel,
            "pos": hist_pos,
        }

        print(f"hist time: {len(hist_time)}")
        print(f"hist vel: {len(hist_vel)}")
        print(f"hist pos: {len(hist_pos)}")

        df = pd.DataFrame(episode_data)
        csv_filename = f"./estimator.csv"
        df.to_csv(csv_filename, index=False)

        # create the data plot
        fig = plt.figure(figsize=(12,20))
        # gs = GridSpec(6, 1, figure=fig)
        gs = GridSpec(10, 1, figure=fig, hspace=.5)
        plt.suptitle(f"Episode PINN Convergence over Time", fontsize=16)

        ax1 = fig.add_subplot(gs[0, 0])

        ax1.plot(hist_total_loss, 'b--', label='True Mass')
        ax1.set(title='Total Loss Over time', xlabel='Time (s)', ylabel='Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # ax1.legend(
        #     loc='center left',          # Anchors the left side of the legend box
        #     bbox_to_anchor=(1.01, 0.5), # Pushes it just outside the right edge of the plot
        #     ncol=1,                     # 1 column looks cleaner when on the side
        #     frameon=True,             
        #     edgecolor='gray',         
        #     fancybox=False,           
        #     shadow=False,             
        #     borderpad=0.5,            
        #     labelspacing=0.3
        # )
        ax1.set_yscale('log')

        fig.savefig(f"./plotting.png")


if __name__ == "__main__":
    main()