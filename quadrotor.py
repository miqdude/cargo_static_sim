import numpy as np
import pybullet as p

class Quadrotor:
    def __init__(self, pybullet_client, robot_id, name, base_link, motor_indices, max_thrust=20.0):
        self.p = pybullet_client
        self.robot_id = robot_id
        self.name = name
        self.base_link = base_link
        self.motor_indices = motor_indices 
        
        # Max thrust per motor in Newtons (Iris motors max out around ~5-7N)
        self.max_thrust = max_thrust 

    def update(self, total_thrust_cmd, torque_cmd):
        """
        total_thrust_cmd: Float, total Newtons of upward force needed.
        torque_cmd: [tau_x, tau_y, tau_z] list or array of desired Roll, Pitch, and Yaw torques (Nm).
        """        
        # ==========================================
        # 2. MOTOR MIXER (Standard 'X' Configuration)
        # ==========================================
        # Calculates exactly how much Newtons each motor must output to satisfy 
        # the requested Thrust + Roll + Pitch + Yaw.
        # Assuming Motor Layout: 
        # 0: Front-Right (CCW), 1: Rear-Left (CCW), 2: Front-Left (CW), 3: Rear-Right (CW)
        
        # ==========================================
        # 1. DRONE PHYSICAL CONSTANTS (From URDF)
        # ==========================================
        Lx = 0.13  # Forward/Back distance (meters)
        Ly = 0.22  # Left/Right distance (meters)
        c = 0.015  # Torque-to-Thrust coefficient
        
        # ==========================================
        # 2. MOTOR MIXER (Asymmetric 'Iris' Configuration)
        # ==========================================
        T_req = total_thrust_cmd / 4.0
        
        # Notice how Roll uses Ly (the left/right distance) 
        # and Pitch uses Lx (the front/back distance)
        roll_req  = torque_cmd[0] / (4.0 * Ly)
        pitch_req = torque_cmd[1] / (4.0 * Lx)
        yaw_req   = torque_cmd[2] / (4.0 * c)

        f0 = T_req - roll_req - pitch_req + yaw_req  # Front-Right (x=0.13, y=-0.22)
        f1 = T_req + roll_req + pitch_req + yaw_req  # Rear-Left   (x=-0.13, y=0.22)
        f2 = T_req + roll_req - pitch_req - yaw_req  # Front-Left  (x=0.13, y=0.22)
        f3 = T_req - roll_req + pitch_req - yaw_req  # Rear-Right  (x=-0.13, y=-0.22)
        
        raw_thrusts = np.array([f0, f1, f2, f3])
        
        # Clip to physical motor limits to prevent physics explosions and negative thrust
        applied_thrusts = np.clip(raw_thrusts, 0.0, self.max_thrust)
        
        # Propeller spin directions for Yaw drag calculation (CCW = 1, CW = -1)
        spin_dirs = [1, 1, -1, -1] 
        
        # ==========================================
        # 3. PYBULLET ACTUATION
        # ==========================================
        for i, motor_idx in enumerate(self.motor_indices):
            
            # A. Apply the Lift Force (Automatically handles Altitude, Roll, and Pitch)
            self.p.applyExternalForce(
                objectUniqueId=self.robot_id,
                linkIndex=motor_idx,
                forceObj=[0, 0, applied_thrusts[i]],
                posObj=[0, 0, 0],
                flags=self.p.LINK_FRAME
            )
            
            # B. Apply the Yaw Drag Torque
            # A propeller spinning CCW creates a CW reaction torque on the drone frame
            reaction_torque = -spin_dirs[i] * c * applied_thrusts[i]
            
            self.p.applyExternalTorque(
                objectUniqueId=self.robot_id,
                linkIndex=motor_idx,
                torqueObj=[0, 0, reaction_torque],
                flags=self.p.LINK_FRAME
            )
            