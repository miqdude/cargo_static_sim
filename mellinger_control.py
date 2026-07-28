import numpy as np
import torch
import torch.nn as nn

class MellingerControl:
    def __init__(self, use_pinn=False, weights_file_path="pinn_cog_weights.pth", mass_total = 8.0):
        self.M_total = mass_total # kg (Mass of rig + 4 drones + 1 Kg Payload)
        self.g = 9.81
        self.dt = 1.0 / 240.0 # PyBullet timestep
        
        # Position Control
        self.kp_pos = np.array([0.8, 0.8, 1.0])  # Dropped significantly
        self.kd_pos = np.array([2.0, 2.0, 3.0])  # Damping is now higher relative to P
        self.ki_pos = np.array([0.1, 0.1, 0.5])  # Lowered integral to prevent slow wind-up
        self.integral_pos_error = np.zeros(3) 
        
        # Attitude Control
        self.kp_att = np.array([10.0, 10.0, 0.2])  # Dropped P gain 
        self.kd_att = np.array([7.0, 7.0, 0.1])  # Kept D high for heavy drag/damping this values gives slow oscilating
        # self.kp_att = np.array([0.8, .8, 0.2])  # Dropped P gain 
        # self.kd_att = np.array([0, 0, 0.1])  # Kept D high for heavy drag/damping this value works

        # 3. Geometric Mounting Positions
        self.r_arms = {
            'iris_1': np.array([ 0.5,  0.5, 0.05]), 
            'iris_2': np.array([ 0.5, -0.5, 0.05]), 
            'iris_3': np.array([-0.5,  0.5, 0.05]), 
            'iris_4': np.array([-0.5, -0.5, 0.05])  
        }
        
        self.c_yaw = [1.0, -1.0, -1.0, 1.0]

    def get_allocation_matrix(self, cog_x=0.0, cog_y=0.0):
        """ 
        Builds the 4x4 Square Matrix around the given System CoG.
        """
        A = np.zeros((4, 4))
        A[0, :] = 1.0 # Total Thrust 
        
        for i, (name, pos) in enumerate(self.r_arms.items()):
            # The effective lever arm is the distance from the motor to the CoG
            eff_x = pos[0] - cog_x
            eff_y = pos[1] - cog_y
            
            A[1, i] = eff_y           # Roll Moment (Y arm)
            A[2, i] = -eff_x          # Pitch Moment (-X arm)
            A[3, i] = self.c_yaw[i]   # Yaw Moment
            
        return A
    
    def _calculate_total_cog(self, m_frame, m_payload, payload_offset):
        """ Internal helper to find the true physical center of rotation """
        m_tot = m_frame + m_payload
        
        # Prevent division by zero
        if m_tot <= 0.0:
            return 0.0, 0.0
            
        cog_x = (m_payload * payload_offset[0]) / m_tot
        cog_y = (m_payload * payload_offset[1]) / m_tot
        return cog_x, cog_y

    def get_follower_commands(
            self,
            payload_offset,
            dt,
            time_now,
            current_pos,
            current_rpy,
            current_vel,
            current_ang_vel,
            current_acc,
            current_ang_acc,
            target_pos,
            target_vel,
            target_ang_vel,
            target_acc,
            target_yaw,
            payload_offset_x,
            payload_offset_y,
            m_frame,
            m_payload,
        ):
        

        # ==========================================
        # Position Control
        # ==========================================

        err_pos = np.array(target_pos) - np.array(current_pos)
        err_vel = np.array(target_vel) - np.array(current_vel)

        self.integral_pos_error += err_pos * dt
        self.integral_pos_error = np.clip(self.integral_pos_error, -5.0, 5.0)

        # r_ddot_des = self.kp_pos * err_pos + self.kd_pos * err_vel + target_acc 
        r_ddot_des = (self.kp_pos * err_pos) + (self.kd_pos * err_vel) + (self.ki_pos * self.integral_pos_error) + target_acc
        
        # ==========================================
        # Hover Controller
        # ==========================================
        
        psi_T = current_rpy[2] # Current Yaw

        phi_des = (1.0 / self.g) * (r_ddot_des[0] * np.sin(psi_T) - r_ddot_des[1] * np.cos(psi_T))
        theta_des = (1.0 / self.g) * (r_ddot_des[0] * np.cos(psi_T) + r_ddot_des[1] * np.sin(psi_T))

        # limiting the target roll and pitch
        # the values is in radians
        # phi_des = np.clip(phi_des, -0.05, 0.05)
        # theta_des = np.clip(theta_des, -0.05, 0.05)

        F_B_des = self.M_total * (r_ddot_des[2] + self.g)

        # ==========================================
        # Attitude Controller
        # ==========================================
        # SAFEGUARD: If target_yaw is passed as a full [roll, pitch, yaw] array, 
        # extract just the Z-axis (yaw) component. Otherwise, use it directly.
        if isinstance(target_yaw, (list, tuple, np.ndarray)):
            yaw_cmd = target_yaw[2]
        else:
            yaw_cmd = target_yaw

        err_att = np.array([
            phi_des - current_rpy[0], 
            theta_des - current_rpy[1], 
            yaw_cmd - current_rpy[2]
        ], dtype=float) # Force a clean float array

        err_ang_rate = np.array(target_ang_vel) - np.array(current_ang_vel)

        # calculate desired torque
        M_des = (self.kp_att * err_att) + (self.kd_att * err_ang_rate)
        M_des[2] = np.clip(M_des[2], -2.0, 2.0) # Clamp yaw

        # ==========================================
        # System Dynamics Allocation
        # ==========================================
        Wrench = np.array([F_B_des, M_des[0], M_des[1], M_des[2]])

        true_cog_x = (1.0 * payload_offset[0]) / self.M_total
        true_cog_y = (1.0 * payload_offset[1]) / self.M_total

        # guess_cog_x = true_cog_x
        # guess_cog_y = true_cog_y

        # Convert the PINN's payload offset guess into the Total System CoG
        guess_cog_x, guess_cog_y = self._calculate_total_cog(m_frame, m_payload, [payload_offset_x, payload_offset_y])

        # Build the dynamic matrix 
        # the controller doesn't know about the shifted CoG by default
        A_final = self.get_allocation_matrix(guess_cog_x, guess_cog_y)
        # A_final = self.get_allocation_matrix(true_cog_x, true_cog_y)
        
        A_inv = np.linalg.pinv(A_final)
        
        # Solve for the exact thrust required by each of the 4 drones
        quad_thrusts = np.dot(A_inv, Wrench)

        # instead of giving the actual wrench
        # the loss must calculate the correct wrench vs the output wrench
        A_perfect = self.get_allocation_matrix(cog_x=true_cog_x, cog_y=true_cog_y)
        correct_wrench = np.dot(A_perfect, quad_thrusts)

        # ---------------------------------------------------------
        # COMMAND PACKAGING
        # ---------------------------------------------------------
        follower_thrust_cmds = {}
        follower_torque_cmds = {}
        
        # Divide the global yaw torque demand equally among the 4 followers
        yaw_torque_per_drone = M_des[2] / 4.0
        
        for i, name in enumerate(self.r_arms.keys()):
            # 1. The Thrust Command 
            # (Differential thrust handles global Altitude, Roll, and Pitch)
            follower_thrust_cmds[name] = max(0.0, quad_thrusts[i])
            
            # 2. The Torque Command
            # Local Roll and Pitch remain 0 so they don't fight the main frame.
            # Local Yaw (Z-axis) is commanded to steer the global heading!
            follower_torque_cmds[name] = np.array([0.0, 0.0, yaw_torque_per_drone])
            
        return follower_thrust_cmds, follower_torque_cmds, Wrench