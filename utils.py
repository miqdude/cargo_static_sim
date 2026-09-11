import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import numpy as np

def get_kinetic_energy(mass, velocity):
    return .5 * mass * velocity**2

def show_evaluation(env, simulation_duration):
    # --- EVALUATION SCRIPT (Runs after the flight is over) ---
    log_errors_3d, log_swing_angles, log_swing_energy = env.get_logging()

    # Convert lists to numpy arrays for fast matrix math
    errors = np.array(log_errors_3d)      # Shape: (N, 3)
    swings = np.array(log_swing_angles)   # Shape: (N,)
    energies = np.array(log_swing_energy) # Shape: (N,)

    # 1. Trajectory Tracking Metrics
    # MAE (Mean Absolute Error) for individual axes
    mae_xyz = np.mean(np.abs(errors), axis=0)

    # RMSE (Root Mean Square Error) for individual axes - Heavily penalizes large spikes!
    rmse_xyz = np.sqrt(np.mean(errors**2, axis=0))

    # 3D Distance Error (The absolute distance between drone and target sphere)
    distances = np.linalg.norm(errors, axis=1)
    mae_3d = np.mean(distances)
    rmse_3d = np.sqrt(np.mean(distances**2))

    # 2. Slosh / Payload Metrics
    # Convert to degrees because reviewers read degrees much easier than radians
    max_swing_deg = np.degrees(np.max(swings))
    mean_swing_deg = np.degrees(np.mean(swings))
    mean_energy = np.mean(energies)

    # 3. Print the Final Report
    print("\n" + "="*45)
    print("       CONTROLLER PERFORMANCE METRICS       ")
    print("="*45)
    print(f"DURATION: {simulation_duration/60} minutes")
    print("="*45)
    print("[TRAJECTORY TRACKING]")
    print(f" 3D Euclidean MAE:    {mae_3d:.4f} m")
    print(f" 3D Euclidean RMSE:   {rmse_3d:.4f} m")
    print(f" -> X-Axis RMSE:      {rmse_xyz[0]:.4f} m")
    print(f" -> Y-Axis RMSE:      {rmse_xyz[1]:.4f} m")
    print(f" -> Z-Axis RMSE:      {rmse_xyz[2]:.4f} m")
    print("-" * 45)
    print("[PAYLOAD STABILITY]")
    print(f" Max Swing Angle:     {max_swing_deg:.2f}°")
    print(f" Mean Swing Angle:    {mean_swing_deg:.2f}°")
    print(f" Mean Slosh Energy:   {mean_energy:.4f} Joules")
    print("="*45 + "\n")

import numpy as np

def get_trajectory(t, path_type="figure8"):
    
    z_start = 1.05
    z_height = 6.0
    z_height_var = 3.0
    
    takeoff_time = 10.0
    hover_time = 20.0 
    
    # --- 1. THE SMOOTH TAKEOFF PHASE (0 to 5 seconds) ---
    if t < takeoff_time:
        s = t / takeoff_time 
        
        scale_pos = 10*(s**3) - 15*(s**4) + 6*(s**5)
        scale_vel = (30*(s**2) - 60*(s**3) + 30*(s**4)) / takeoff_time
        scale_acc = (60*(s) - 180*(s**2) + 120*(s**3)) / (takeoff_time**2)
        
        current_z = z_start + ((z_height - z_start) * scale_pos)
        current_vz = (z_height - z_start) * scale_vel
        current_az = (z_height - z_start) * scale_acc
        
        pos = [0.0, 0.0, current_z]
        vel = [0.0, 0.0, current_vz]
        acc = [0.0, 0.0, current_az]
        
    # --- 2. THE STABILIZATION PHASE (5 to 10 seconds) ---
    elif t < hover_time:
        pos = [0.0, 0.0, z_height]
        vel = [0.0, 0.0, 0.0]
        acc = [0.0, 0.0, 0.0]
    
    # --- 3. THE FLIGHT PHASE (t > 10 seconds) ---
    else:
        flight_t = t - hover_time
        
        if path_type == "circle":
            radius = 10.0
            omega = 0.15 
        
            pos = [radius * np.sin(omega * flight_t), 
                   radius * (1 - np.cos(omega * flight_t)), 
                   z_height + (z_height_var * np.sin(omega * flight_t))]
                   
            vel = [radius * omega * np.cos(omega * flight_t), 
                   radius * omega * np.sin(omega * flight_t), 
                   z_height_var * omega * np.cos(omega * flight_t)]
                   
            acc = [-radius * (omega**2) * np.sin(omega * flight_t), 
                    radius * (omega**2) * np.cos(omega * flight_t), 
                   -z_height_var * (omega**2) * np.sin(omega * flight_t)]
            
        elif path_type == "figure8":
            radius = 10.0
            omega = 0.10 
            
            pos = [radius * np.sin(omega * flight_t), 
                   radius * np.sin(omega * flight_t) * np.cos(omega * flight_t), 
                   z_height + (z_height_var * np.sin(omega * flight_t))]
            
            vel = [radius * omega * np.cos(omega * flight_t),
                   radius * omega * (np.cos(omega * flight_t)**2 - np.sin(omega * flight_t)**2),
                   z_height_var * omega * np.cos(omega * flight_t)]
                   
            acc = [-radius * (omega**2) * np.sin(omega * flight_t),
                   -4.0 * radius * (omega**2) * np.sin(omega * flight_t) * np.cos(omega * flight_t),
                   -z_height_var * (omega**2) * np.sin(omega * flight_t)]
            
        elif path_type == "hover":
            pos = [0.0, 0.0, z_height]
            vel = [0.0, 0.0, 0.0]
            acc = [0.0, 0.0, 0.0]

        elif path_type == "hover_yaw":
            z_height = 5.0 # Or whatever your default hover altitude is
            
            # 1. Position remains perfectly static
            pos = [0.0, 0.0, z_height]
            vel = [0.0, 0.0, 0.0]
            acc = [0.0, 0.0, 0.0]
            
            # 2. Yaw Sweep Parameters for Persistent Excitation
            amplitude = np.pi / 2.0  # Sweep 90 degrees left and right
            frequency = 0.3          # How fast it sweeps back and forth
            
            # 3. Calculate current angle and angular velocity
            yaw = amplitude * np.sin(2.0 * np.pi * frequency * t)
            yaw_rate = amplitude * 2.0 * np.pi * frequency * np.cos(2.0 * np.pi * frequency * t)
            
            # 4. Pack into the exact arrays expected by the flight loop
            rpy = [0.0, 0.0, yaw]
            ang_vel = [0.0, 0.0, yaw_rate] 
            
        elif path_type == "forward":
            speed = 0.7  
            
            pos = [speed * flight_t, 0.0, z_height]
            vel = [speed, 0.0, 0.0]
            acc = [0.0, 0.0, 0.0]

        elif path_type == "figure_eight":
            # Frequencies and Amplitudes
            A_x, w_x = 1.0, 0.5  # 1m sweep, slow
            A_y, w_y = 1.0, 1.0  # 1m sweep, fast
            A_z, w_z = 0.5, 0.2  # 0.5m vertical bob (scaled down to match)
            
            # 1. POSITION
            pos = [
                A_x * np.sin(w_x * flight_t),
                A_y * np.sin(w_y * flight_t),
                z_height + A_z * np.sin(w_z * flight_t)
            ]
            
            # 2. VELOCITY (First Derivative)
            vel = [
                A_x * w_x * np.cos(w_x * flight_t),
                A_y * w_y * np.cos(w_y * flight_t),
                A_z * w_z * np.cos(w_z * flight_t)
            ]
            
            # 3. ACCELERATION (Second Derivative)
            acc = [
                -A_x * (w_x**2) * np.sin(w_x * flight_t),
                -A_y * (w_y**2) * np.sin(w_y * flight_t),
                -A_z * (w_z**2) * np.sin(w_z * flight_t)
            ]
            
            # 4. YAW AND ANGULAR VELOCITY
            # We actively sweep the yaw back and forth by 45 degrees (0.78 rad)
            # This explicitly excites the J_zz (Yaw Moment of Inertia)
            A_yaw, w_yaw = 0.78, 0.5
            target_yaw = A_yaw * np.sin(w_yaw * flight_t)
            
            # target_ang_vel expects [roll_dot, pitch_dot, yaw_dot]
            # Roll and Pitch dots are implicitly handled by the position controller
            ang_vel = [0.0, 0.0, A_yaw * w_yaw * np.cos(w_yaw * flight_t)]
            
            # return target_pos, target_vel, target_ang_vel, target_acc, target_yaw

        elif path_type == "figure_eight_3d":
            # ---------------------------------------------------------
            # PARAMETERS (Tune these to change the shape and speed)
            # ---------------------------------------------------------
            z_hover = 5.0      # Base altitude
            
            A_x = 3.0          # How wide the 8 is (Meters)
            A_y = 3.0          # How long the 8 is (Meters)
            A_z = 1.5          # How high/low it dips (Meters)
            
            f_x = 0.05         # Base speed (Hz)
            f_y = 2.0 * f_x    # Y frequency MUST be double X to form the 8
            f_z = f_x          # Z dips in sync with the loops
            
            # Convert frequencies to angular velocities (omega)
            w_x = 2.0 * np.pi * f_x
            w_y = 2.0 * np.pi * f_y
            w_z = 2.0 * np.pi * f_z
            
            # ---------------------------------------------------------
            # 1. POSITION (The mathematical path)
            # ---------------------------------------------------------
            x = A_x * np.sin(w_x * t)
            y = A_y * np.sin(w_y * t)
            z = z_hover + A_z * np.sin(w_z * t)
            pos = [x, y, z]
            
            # ---------------------------------------------------------
            # 2. VELOCITY (First Derivative)
            # ---------------------------------------------------------
            vx = A_x * w_x * np.cos(w_x * t)
            vy = A_y * w_y * np.cos(w_y * t)
            vz = A_z * w_z * np.cos(w_z * t)
            vel = [vx, vy, vz]
            
            # ---------------------------------------------------------
            # 3. ACCELERATION (Second Derivative)
            # ---------------------------------------------------------
            ax = -A_x * (w_x**2) * np.sin(w_x * t)
            ay = -A_y * (w_y**2) * np.sin(w_y * t)
            az = -A_z * (w_z**2) * np.sin(w_z * t)
            acc = [ax, ay, az]
            
            # ---------------------------------------------------------
            # 4. YAW ALIGNMENT (Look where it's going)
            # ---------------------------------------------------------
            # Calculate heading dynamically based on velocity vector
            target_yaw = np.arctan2(vy, vx)
            rpy = [0.0, 0.0, target_yaw]
            
            # Angular velocity (Derivative of arctan2)
            yaw_rate = (vx * ay - vy * ax) / (vx**2 + vy**2 + 1e-6)
            ang_vel = [0.0, 0.0, yaw_rate]
            
            # Ensure your function returns exactly these 5 variables in this order!
            # return pos, vel, acc, rpy, ang_vel

        elif path_type == "figure_eight_3d_rotating":
            # ---------------------------------------------------------
            # PARAMETERS 
            # ---------------------------------------------------------
            z_hover = 5.0      
            
            A_x = 3.0          
            A_y = 3.0          
            A_z = 1.5          
            
            f_x = 0.05         # Drone completes 1 full lap every 20 seconds
            f_y = 2.0 * f_x    
            f_z = f_x          
            
            # ---> NEW: Path Rotation Speed
            # How fast the "8" itself rotates. 
            # E.g., f_x / 4.0 means the path completes one full rotation every 4 laps
            f_path = f_x / 4.0 
            
            w_x = 2.0 * np.pi * f_x
            w_y = 2.0 * np.pi * f_y
            w_z = 2.0 * np.pi * f_z
            w_path = 2.0 * np.pi * f_path # Path angular velocity
            
            # ---------------------------------------------------------
            # 1. BASE KINEMATICS (Static Path)
            # ---------------------------------------------------------
            # Position
            bx = A_x * np.sin(w_x * t)
            by = A_y * np.sin(w_y * t)
            bz = z_hover + A_z * np.sin(w_z * t)
            
            # Velocity
            bvx = A_x * w_x * np.cos(w_x * t)
            bvy = A_y * w_y * np.cos(w_y * t)
            bvz = A_z * w_z * np.cos(w_z * t)
            
            # Acceleration
            bax = -A_x * (w_x**2) * np.sin(w_x * t)
            bay = -A_y * (w_y**2) * np.sin(w_y * t)
            baz = -A_z * (w_z**2) * np.sin(w_z * t)
            
            # ---------------------------------------------------------
            # 2. APPLY CONTINUOUS Z-AXIS ROTATION
            # ---------------------------------------------------------
            theta = w_path * t
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # Rotate Position
            x = bx * cos_t - by * sin_t
            y = bx * sin_t + by * cos_t
            z = bz
            pos = [x, y, z]
            
            # Rotate Velocity (Approximation for slow path rotation)
            vx = bvx * cos_t - bvy * sin_t
            vy = bvx * sin_t + bvy * cos_t
            vz = bvz
            vel = [vx, vy, vz]
            
            # Rotate Acceleration (Approximation for slow path rotation)
            ax = bax * cos_t - bay * sin_t
            ay = bax * sin_t + bay * cos_t
            az = baz
            acc = [ax, ay, az]
            
            # ---------------------------------------------------------
            # 3. YAW ALIGNMENT 
            # ---------------------------------------------------------
            target_yaw = np.arctan2(vy, vx)
            rpy = [0.0, 0.0, target_yaw]
            
            yaw_rate = (vx * ay - vy * ax) / (vx**2 + vy**2 + 1e-6)
            ang_vel = [0.0, 0.0, yaw_rate]
            
            # return pos, vel, acc, rpy, ang_vel

        elif "lawn":
            # 1. Timing and Amplitudes
            T_cycle = 20.0       # Total time for one full Forward-Backward-Left-Right cycle
            T_phase = 5.0        # Time for each individual movement
            A = 2.0              # Max distance in meters (drone will move 0 -> 2m -> 0)
            z_center = 5.0       # Constant hover altitude
            
            # Frequency for the cosine wave to complete one full cycle in T_phase
            w = (2 * np.pi) / T_phase 
            
            # Find where we are in the current cycle
            # t = t % T_cycle
            
            # Initialize zero arrays
            pos = np.array([0.0, 0.0, z_center])
            vel = np.array([0.0, 0.0, 0.0])
            acc = np.array([0.0, 0.0, 0.0])
            
            pos[0] = (A / 2.0) * (1 - np.cos(w * t))
            vel[0] = (A * w / 2.0) * np.sin(w * t)
            acc[0] = (A * w**2 / 2.0) * np.cos(w * t)

        elif "lemiscate":    
            # 1. Tuning Parameters
            A_x = 3.0  # Max distance forward/backward (meters)
            A_y = 1.5  # Max distance left/right (meters)
            w = 0.5    # Base angular frequency (rad/s) - controls the speed
            z_center = 5.0 # Constant hover altitude
            
            # 2. Position (Flat Outputs)
            pos = np.array([
                A_x * np.sin(w * t),
                A_y * np.sin(2 * w * t),
                z_center
            ])
            
            # 3. Velocity (First Derivative)
            vel = np.array([
                A_x * w * np.cos(w * t),
                2 * A_y * w * np.cos(2 * w * t),
                0.0
            ])
            
            # 4. Acceleration (Second Derivative)
            acc = np.array([
                -A_x * (w**2) * np.sin(w * t),
                -4 * A_y * (w**2) * np.sin(2 * w * t),
                0.0
            ])
            
            # 5. Yaw and Angular Velocity
            # Keeping yaw constant at 0 so the frame always faces "forward"
            target_rpy = np.array([0.0, 0.0, 0.0])
            target_ang_vel = np.array([0.0, 0.0, 0.0])
            
            return pos, vel, acc, target_rpy, target_ang_vel

    # # ==========================================
    # # NEW: ORIENTATION (GHOST CAR) CALCULATIONS
    # # ==========================================
    # # We want the ghost car to point its nose in the direction it is moving.
    # # We only calculate this if the drone is actually moving horizontally.
    # speed_sq = vel[0]**2 + vel[1]**2
    
    # if speed_sq > 1e-6:
    #     # Yaw is the angle of the velocity vector
    #     target_yaw = np.arctan2(vel[1], vel[0])
        
    #     # Yaw rate (angular velocity) is the mathematical derivative of arctan2
    #     # Equation: (vx * ay - vy * ax) / (vx^2 + vy^2)
    #     target_yaw_rate = (vel[0] * acc[1] - vel[1] * acc[0]) / speed_sq
    # else:
    #     # If hovering or going straight up, stay locked forward
    #     target_yaw = 0.0
    #     target_yaw_rate = 0.0
        
    # The target Roll and Pitch are 0.0 (the trajectory stays flat), only Yaw changes
    rpy = [0.0, 0.0, 0]
    
    # Angular velocity is purely around the Z-axis (Yaw rate)
    ang_vel = [0.0, 0.0, 0]
            
    return pos, vel, acc, rpy, ang_vel

def calculate_total_cog(m_frame, m_payload, payload_offset):
    """
    Calculates the new total system CoG using standard numbers.
    
    m_frame: Mass of the drone frame (kg)
    m_payload: Mass of the payload (kg)
    payload_offset: List or tuple of [X, Y, Z] payload position (meters)
    """
    total_mass = m_frame + m_payload
    
    # Weighted average: (m_p * offset) / m_total
    system_cog_x = (m_payload * payload_offset[0]) / total_mass
    system_cog_y = (m_payload * payload_offset[1]) / total_mass
    system_cog_z = (m_payload * payload_offset[2]) / total_mass
    
    return [system_cog_x, system_cog_y, system_cog_z]

def calculate_payload_offset(m_frame, m_payload, total_cog):
    """
    Converts the Total System CoG back into the raw Payload X, Y, Z offset.
    
    m_frame: Mass of the empty drone frame (kg)
    m_payload: Estimated or known mass of the payload (kg)
    total_cog: List or tuple of [X, Y, Z] Total System CoG (meters)
    """
    # Prevent division by zero if the payload mass drops to zero
    if m_payload <= 0.001:
        return [0.0, 0.0, 0.0]
        
    total_mass = m_frame + m_payload
    mass_ratio = total_mass / m_payload
    
    offset_x = total_cog[0] * mass_ratio
    offset_y = total_cog[1] * mass_ratio
    offset_z = total_cog[2] * mass_ratio
    
    return [offset_x, offset_y, offset_z]

def get_3d_frame_coordinates(pos, rpy, estimated_cog_x, estimated_cog_y):
    """
    Transforms the local drone arms and payload offset into Global 3D coordinates.
    """
    roll, pitch, yaw = rpy
    
    # Standard Aerospace Rotation Matrix (Z-Y-X)
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_matrix = R_z @ R_y @ R_x
    
    # 1. Define the local arms (Matching your Mellinger geometric parameters)
    # Diagonal 1: from Iris_4 to Iris_1
    arm1_local_start = np.array([-0.5, -0.5, 0.05])
    arm1_local_end   = np.array([ 0.5,  0.5, 0.05])
    
    # Diagonal 2: from Iris_3 to Iris_2
    arm2_local_start = np.array([-0.5,  0.5, 0.05])
    arm2_local_end   = np.array([ 0.5, -0.5, 0.05])
    
    # Local Payload Position (estimated by PINN)
    payload_local = np.array([estimated_cog_x, estimated_cog_y, 0.0])

    # 2. Rotate and Translate to Global Coordinates
    arm1_global_start = pos + R_matrix @ arm1_local_start
    arm1_global_end   = pos + R_matrix @ arm1_local_end
    
    arm2_global_start = pos + R_matrix @ arm2_local_start
    arm2_global_end   = pos + R_matrix @ arm2_local_end
    
    payload_global = pos + R_matrix @ payload_local

    return arm1_global_start, arm1_global_end, arm2_global_start, arm2_global_end, payload_global


def animate_episode_flight(plt, episode_num, ep_actual_x, ep_actual_y, ep_actual_z, ep_actual_rpy, 
                           ep_target_x, ep_target_y, ep_target_z, 
                           pred_system_cog_x, pred_system_cog_y, 
                           frame_skip=10):
    """
    Creates a smooth 3D animation of the drone's flight path and physical tilt.
    frame_skip: Renders every Nth frame to prevent Jupyter from crashing on large arrays.
    """
    print("Rendering 3D Flight Animation... Please wait.")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set(title=f"Replay Flight trajectory {episode_num}", xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)')
    
    # 1. Setup Static Elements (The Target Path)
    ax.plot(ep_target_x, ep_target_y, ep_target_z, label='Target Path', color='gray', linestyle='--', alpha=0.6)
    
    # 2. Setup Dynamic Elements (The lines we will update every frame)
    line_actual_traj, = ax.plot([], [], [], label='Actual Flight Path', color='purple', linewidth=2)
    line_frame_arm1, = ax.plot([], [], [], color='black', linewidth=3, label='Rigid Frame')
    line_frame_arm2, = ax.plot([], [], [], color='black', linewidth=3)
    scatter_payload, = ax.plot([], [], [], marker='o', color='red', markersize=8, linestyle='None', label='Est. Payload CoG')
    
    ax.legend()
    
    # 3. Fix the Camera Boundaries (So it doesn't bounce around)
    max_range = np.array([
        max(ep_actual_x) - min(ep_actual_x),
        max(ep_actual_y) - min(ep_actual_y),
        max(ep_actual_z) - min(ep_actual_z)
    ]).max() / 2.0

    mid_x = (max(ep_actual_x) + min(ep_actual_x)) * 0.5
    mid_y = (max(ep_actual_y) + min(ep_actual_y)) * 0.5
    mid_z = (max(ep_actual_z) + min(ep_actual_z)) * 0.5
    
    ax.set_xlim(mid_x - max_range - 0.5, mid_x + max_range + 0.5)
    ax.set_ylim(mid_y - max_range - 0.5, mid_y + max_range + 0.5)
    ax.set_zlim(mid_z - max_range - 0.5, mid_z + max_range + 0.5)
    ax.set_box_aspect((1, 1, 1)) 

    # Subsample the arrays for smooth ~30fps playback
    frames_to_render = range(0, len(ep_actual_x), frame_skip)

    # 4. The Update Function (Called once per frame)
    def update(frame_idx):
        # Draw the purple trail up to the current frame
        line_actual_traj.set_data(ep_actual_x[:frame_idx], ep_actual_y[:frame_idx])
        line_actual_traj.set_3d_properties(ep_actual_z[:frame_idx])
        
        # Get the drone's position and orientation AT this exact frame
        pos = np.array([ep_actual_x[frame_idx], ep_actual_y[frame_idx], ep_actual_z[frame_idx]])
        rpy = ep_actual_rpy[frame_idx]
        
        # Calculate the 3D wireframe using your existing transformation function
        a1_s, a1_e, a2_s, a2_e, p_glob = get_3d_frame_coordinates(
            pos, rpy, estimated_cog_x=pred_system_cog_x, estimated_cog_y=pred_system_cog_y
        )
        
        # Draw the drone arms and payload
        line_frame_arm1.set_data([a1_s[0], a1_e[0]], [a1_s[1], a1_e[1]])
        line_frame_arm1.set_3d_properties([a1_s[2], a1_e[2]])
        
        line_frame_arm2.set_data([a2_s[0], a2_e[0]], [a2_s[1], a2_e[1]])
        line_frame_arm2.set_3d_properties([a2_s[2], a2_e[2]])
        
        scatter_payload.set_data([p_glob[0]], [p_glob[1]])
        scatter_payload.set_3d_properties([p_glob[2]])
        
        return line_actual_traj, line_frame_arm1, line_frame_arm2, scatter_payload

    # Compile the animation
    anim = FuncAnimation(fig, update, frames=frames_to_render, interval=30, blit=False)

    print(f"Saving animation to flight_replay_ep_{episode_num}.mp4...")
    anim.save(f'./flight_replay_ep_{episode_num}_.mp4', fps=24)
    print("Done!")
    
    # Close the static plot so it doesn't print twice
    plt.close(fig) 
    
    # Display as an interactive HTML5 video right inside Jupyter
    # return HTML(anim.to_jshtml())

def calculate_true_principal_inertia(base_J_diag, payload_mass, payload_offset):
    """
    Calculates the exact True Inertia of the multi-rotor system using the Parallel Axis Theorem.
    Assumes the payload is a dense point mass located at the offset.
    """
    px, py, pz = payload_offset
    
    # Calculate the shift in inertia caused by the offset mass
    delta_Jxx = payload_mass * (py**2 + pz**2)
    delta_Jyy = payload_mass * (px**2 + pz**2)
    delta_Jzz = payload_mass * (px**2 + py**2)
    
    true_Jxx = base_J_diag[0] + delta_Jxx
    true_Jyy = base_J_diag[1] + delta_Jyy
    true_Jzz = base_J_diag[2] + delta_Jzz

    return true_Jxx, true_Jyy, true_Jzz