import pybullet as p
import pybullet_data
import time
import math

# --- 1. Physics Engine Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("final_assembly.urdf", basePosition=[0, 0, 0.5])

# --- 2. Rotor Mapping ---
diag_1_rotors = [] # Top-Right (iris_1) and Bottom-Left (iris_4)
diag_2_rotors = [] # Bottom-Right (iris_2) and Top-Left (iris_3)

num_joints = p.getNumJoints(robot_id)
for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    link_name = joint_info[12].decode('utf-8')
    
    if "rotor" in link_name:
        if "iris_1" in link_name or "iris_4" in link_name:
            diag_1_rotors.append(i)
        elif "iris_2" in link_name or "iris_3" in link_name:
            diag_2_rotors.append(i)

# --- 3. Physics Parameters & Thrust Calculations ---
mass = 7.8  # kg
gravity = 9.81  # m/s^2
num_rotors = 16

# Calculate base thrust required to counteract gravity
total_hover_thrust = mass * gravity
z_thrust_per_rotor = total_hover_thrust / num_rotors

# Account for the 11-degree inward tilt
tilt_angle_rad = math.radians(11)
base_thrust_vector = z_thrust_per_rotor / math.cos(tilt_angle_rad)

# Yaw Differential magnitude
target_yaw_offset = 0.5  # Newtons

# --- 4. Controller & Timer Setup ---
target_altitude = 3.0  
kp_z = 2.0             
kd_z = 1.5             

dt = 1.0 / 240.0
sim_time = 0.0

hover_achieved = False
time_hover_achieved = 0.0
current_state = "TAKEOFF"

print("Taking off to 3 meters...")

# --- 5. Physics Loop ---
try:
    while True:
        # Get current state
        pos, ori = p.getBasePositionAndOrientation(robot_id)
        current_z = pos[2]
        
        linear_vel, angular_vel = p.getBaseVelocity(robot_id)
        current_vz = linear_vel[2]
        
        # Calculate altitude error (P) and velocity (D)
        error_z = target_altitude - current_z
        z_correction = (kp_z * error_z) - (kd_z * current_vz)
        thrust_correction_per_rotor = z_correction / num_rotors
        
        # --- YAW STATE MACHINE ---
        if not hover_achieved and abs(error_z) < 0.1:
            hover_achieved = True
            time_hover_achieved = sim_time
            current_state = "HOVER_WAIT"
            print(f"[{sim_time:.2f}s] Hover achieved! Waiting 4 seconds...")

        current_yaw_offset = 0.0
        
        if hover_achieved:
            t_elapsed = sim_time - time_hover_achieved
            
            # Start maneuvers after the initial 4-second wait
            if t_elapsed >= 4.0:
                cycle_time = t_elapsed - 4.0
                
                # Switch direction every 4 seconds using modulo arithmetic
                phase = int(cycle_time / 10.0) % 2 
                
                if phase == 0:
                    if current_state != "YAW_CW":
                        current_state = "YAW_CW"
                        print(f"[{sim_time:.2f}s] Engaging Yaw CLOCKWISE.")
                    # Positive offset spins CW
                    current_yaw_offset = target_yaw_offset 
                else:
                    if current_state != "YAW_CCW":
                        current_state = "YAW_CCW"
                        print(f"[{sim_time:.2f}s] Engaging Yaw COUNTER-CLOCKWISE.")
                    # Negative offset spins CCW by giving Diag 2 more thrust
                    current_yaw_offset = -target_yaw_offset

        # --- APPLY THRUST ---
        thrust_diag_1 = base_thrust_vector + thrust_correction_per_rotor + current_yaw_offset
        thrust_diag_2 = base_thrust_vector + thrust_correction_per_rotor - current_yaw_offset
        
        # Ensure motors never spin backwards
        thrust_diag_1 = max(0.0, thrust_diag_1)
        thrust_diag_2 = max(0.0, thrust_diag_2)

        # Apply to Diagonal 1
        for rotor_idx in diag_1_rotors:
            p.applyExternalForce(
                objectUniqueId=robot_id,
                linkIndex=rotor_idx,
                forceObj=[0, 0, thrust_diag_1],  
                posObj=[0, 0, 0],                   
                flags=p.LINK_FRAME  
            )
            
        # Apply to Diagonal 2
        for rotor_idx in diag_2_rotors:
            p.applyExternalForce(
                objectUniqueId=robot_id,
                linkIndex=rotor_idx,
                forceObj=[0, 0, thrust_diag_2],  
                posObj=[0, 0, 0],                   
                flags=p.LINK_FRAME 
            )
        
        p.stepSimulation()
        sim_time += dt
        time.sleep(dt)

except KeyboardInterrupt:
    print("\nSimulation stopped.")
    p.disconnect()