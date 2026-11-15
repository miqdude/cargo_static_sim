import numpy as np
import torch
import torch.nn as nn
import math

class MellingerControl:
    def __init__(self, mass_total=8.0):
        self.M_total = mass_total # kg (Mass of rig + 4 drones + 1 Kg Payload)
        self.g = 9.81
        self.dt = 1.0 / 240.0 # PyBullet timestep
        
        # 11-Degree Inward Tilt Geometry
        self.tilt_angle_rad = math.radians(11.0)
        self.cos_g = np.cos(self.tilt_angle_rad)
        self.sin_g = np.sin(self.tilt_angle_rad)
        
        # Position Control
        self.kp_pos = np.array([0.8, 0.8, 1.0])  
        self.kd_pos = np.array([2.0, 2.0, 3.0])  
        self.ki_pos = np.array([0.1, 0.1, 0.5])  
        self.integral_pos_error = np.zeros(3) 
        
        # Attitude Control
        self.kp_att = np.array([10.0, 10.0, 5.0])  
        self.kd_att = np.array([7.0, 7.0, 2.5])    

        # 3. Geometric Mounting Positions (x, y, z relative to frame center)
        self.r_arms = {
            'iris_1': np.array([ 0.5,  0.5, 0.0]), 
            'iris_2': np.array([ 0.5, -0.5, 0.0]), 
            'iris_3': np.array([-0.5,  0.5, 0.0]), 
            'iris_4': np.array([-0.5, -0.5, 0.0])  
        }

    def get_allocation_matrix(self, cog_x=0.0, cog_y=0.0):
        """ 
        Builds the 4x4 Square Matrix around the given System CoG, 
        accounting for the 11-degree inward tilt of the rotors.
        """
        A = np.zeros((4, 4))
        
        for i, (name, pos) in enumerate(self.r_arms.items()):
            eff_x = pos[0] - cog_x
            eff_y = pos[1] - cog_y
            eff_z = pos[2] # 0.05m mounting height
            
            # Determine inward tilt direction based on URDF pitching
            # iris_1 & 2 (x > 0) are pitched to thrust towards -X (so thrust has +X component)
            # iris_3 & 4 (x < 0) are pitched to thrust towards +X (so thrust has -X component)
            s_i = -1.0 if pos[0] > 0 else 1.0
            
            # Row 0: Total Z Thrust (Only the vertical component lifts the frame)
            A[0, i] = self.cos_g
            
            # Row 1: Roll Moment (Mx = r_y*F_z - r_z*F_y)
            A[1, i] = eff_y * self.cos_g
            
            # Row 2: Pitch Moment (My = r_z*F_x - r_x*F_z)
            # The inward horizontal force (F_x) acting at height (eff_z) creates a slight pitch coupling!
            A[2, i] = (eff_z * s_i * self.sin_g) - (eff_x * self.cos_g)
            
            # Row 3: Yaw Moment (Mz = r_x*F_y - r_y*F_x)
            # YAW FIX: This is now handled entirely by the horizontal component of the tilted thrust!
            A[3, i] = -eff_y * s_i * self.sin_g
            
        return A

    def _calculate_total_cog(self, m_frame, m_payload, payload_offset):
        """ Internal helper to find the true physical center of rotation """
        m_tot = m_frame + m_payload
        if m_tot <= 0.0:
            return 0.0, 0.0
            
        cog_x = (m_payload * payload_offset[0]) / m_tot
        cog_y = (m_payload * payload_offset[1]) / m_tot
        return cog_x, cog_y

    def get_follower_commands(
            self,
            payload_offset,
            dt,
            current_pos,
            current_rpy,
            current_vel,
            current_ang_vel,
            target_pos,
            target_vel,
            target_ang_vel,
            target_acc,
            target_yaw,
            m_frame,
            m_payload,
        ):

        # ==========================================
        # Position Control
        # ==========================================
        err_pos = np.array(target_pos) - np.array(current_pos)
        err_vel = np.array(target_vel) - np.array(current_vel)

        self.integral_pos_error += err_pos * dt
        self.integral_pos_error = np.clip(self.integral_pos_error, -20.0, 20.0)

        r_ddot_des = (self.kp_pos * err_pos) + (self.kd_pos * err_vel) + (self.ki_pos * self.integral_pos_error) + target_acc

        # ==========================================
        # Hover Controller
        # ==========================================
        psi_T = current_rpy[2] # Current Yaw
        
        cos_psi = np.cos(psi_T)
        sin_psi = np.sin(psi_T)
        
        R_world_to_body = np.array([
            [ cos_psi,  sin_psi, 0.0],
            [-sin_psi,  cos_psi, 0.0],
            [     0.0,      0.0, 1.0]
        ])

        phi_des = (1.0 / self.g) * (r_ddot_des[0] * np.sin(psi_T) - r_ddot_des[1] * np.cos(psi_T))
        theta_des = (1.0 / self.g) * (r_ddot_des[0] * np.cos(psi_T) + r_ddot_des[1] * np.sin(psi_T))

        F_B_des = self.M_total * (r_ddot_des[2] + self.g)

        # ==========================================
        # Attitude Controller
        # ==========================================
        if isinstance(target_yaw, (list, tuple, np.ndarray)):
            yaw_cmd = target_yaw[2]
        else:
            yaw_cmd = target_yaw

        raw_yaw_error = yaw_cmd - current_rpy[2]
        wrapped_yaw_error = np.arctan2(np.sin(raw_yaw_error), np.cos(raw_yaw_error))

        err_att = np.array([
            phi_des - current_rpy[0], 
            theta_des - current_rpy[1], 
            wrapped_yaw_error
        ], dtype=float)

        body_ang_vel = np.dot(R_world_to_body, current_ang_vel)
        err_ang_rate = np.array(target_ang_vel) - body_ang_vel

        M_des = (self.kp_att * err_att) + (self.kd_att * err_ang_rate)
        M_des[2] = np.clip(M_des[2], -1.5, 1.5) # Clamp yaw

        # ==========================================
        # System Dynamics Allocation
        # ==========================================
        Wrench = [F_B_des, M_des[0], M_des[1], M_des[2]]

        guess_cog_x, guess_cog_y = self._calculate_total_cog(m_frame, m_payload, [payload_offset[0], payload_offset[1]])

        A_final = self.get_allocation_matrix(guess_cog_x, guess_cog_y)
        A_inv = np.linalg.pinv(A_final)
        
        quad_thrusts = np.dot(A_inv, Wrench)

        # ---------------------------------------------------------
        # COMMAND PACKAGING
        # ---------------------------------------------------------
        follower_thrust_cmds = {}
        follower_torque_cmds = {}
        
        for i, name in enumerate(self.r_arms.keys()):
            follower_thrust_cmds[name] = float(max(0.0, quad_thrusts[i]))
            
            # The matrix automatically converted the yaw demand (M_des[2]) into differential thrust.
            # We NO LONGER need to command mechanical torque to the local drones!
            follower_torque_cmds[name] = [0.0, 0.0, 0.0]
            
        return follower_thrust_cmds, follower_torque_cmds, Wrench