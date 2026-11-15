import numpy as np
import math

class TestController:
    def __init__(self, mass_total=8.0):
        self.M_total = mass_total
        self.g = 9.81
        
        # 11-Degree Tilt Geometry
        self.tilt_angle_rad = math.radians(11.0)
        self.cos_g = np.cos(self.tilt_angle_rad)
        
        # 1. Position Feedback PID (X, Y, Z)
        self.kp_pos = np.array([1.2, 1.2, 15.0])  
        self.kd_pos = np.array([1.5, 1.5, 10.0])  
        
        # 2. Attitude Feedback PID (Roll, Pitch, Yaw)
        self.kp_att = np.array([10.0, 10.0, 5.0])
        self.kd_att = np.array([5.0, 5.0, 2.5])
        
        self.r_arms = {
            'iris_1': np.array([ 0.5,  0.5, 0.0]), # Front-Left
            'iris_2': np.array([ 0.5, -0.5, 0.0]), # Front-Right
            'iris_3': np.array([-0.5,  0.5, 0.0]), # Back-Left
            'iris_4': np.array([-0.5, -0.5, 0.0])  # Back-Right
        }

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
        # 1. Position Outer-Loop (X, Y, Z)
        # ==========================================
        err_pos = np.array(target_pos) - np.array(current_pos)
        err_vel = np.array(target_vel) - np.array(current_vel)
        
        a_des = (self.kp_pos * err_pos) + (self.kd_pos * err_vel)
        
        # Total required Z thrust before allocation
        total_z_thrust = self.M_total * (self.g + a_des[2])
        
        # ==========================================
        # 2. Attitude Conversion (Acceleration -> Roll/Pitch)
        # ==========================================
        current_yaw = current_rpy[2]
        
        phi_des = (1.0 / self.g) * (a_des[0] * np.sin(current_yaw) - a_des[1] * np.cos(current_yaw))
        theta_des = (1.0 / self.g) * (a_des[0] * np.cos(current_yaw) + a_des[1] * np.sin(current_yaw))
        
        phi_des = np.clip(phi_des, -0.5, 0.5)
        theta_des = np.clip(theta_des, -0.5, 0.5)

        # ==========================================
        # 3. Attitude Inner-Loop (Roll, Pitch, Yaw)
        # ==========================================
        if isinstance(target_yaw, (list, tuple, np.ndarray)):
            yaw_cmd = target_yaw[2]
        else:
            yaw_cmd = target_yaw
            
        raw_yaw_error = yaw_cmd - current_yaw
        yaw_error = np.arctan2(np.sin(raw_yaw_error), np.cos(raw_yaw_error))
        
        err_att = np.array([
            phi_des - current_rpy[0],
            theta_des - current_rpy[1],
            yaw_error
        ])
        
        if isinstance(target_ang_vel, (list, tuple, np.ndarray)):
            yaw_rate_cmd = target_ang_vel[2]
        else:
            yaw_rate_cmd = 0.0
            
        err_rate = np.array([0.0, 0.0, yaw_rate_cmd]) - np.array(current_ang_vel)
        
        torques = (self.kp_att * err_att) + (self.kd_att * err_rate)
        
        torque_roll = torques[0]
        torque_pitch = torques[1]
        torque_yaw = np.clip(torques[2], -3.0, 3.0)

        # ==========================================
        # 4. Explicit Allocation Matrix (From Paper Eq 13 & 14)
        # ==========================================
        # Mapping the paper's numeric subscripts (1-4) to our physical URDF positions[cite: 2]:
        # Based on the signs in Eq 14, T1 and T3 pitch up (+My), while T2 and T4 pitch down (-My).
        # Therefore, T1 and T3 are the rear drones; T2 and T4 are the front drones.
        # - T1: Back-Right  (iris_4)
        # - T2: Front-Left  (iris_1)
        # - T3: Back-Left   (iris_3)
        # - T4: Front-Right (iris_2)
        # - T5: Central UAV (Excluded from this 4-drone implementation)[cite: 2]
        
        # Calculate geometric moment arms 
        l = np.sqrt(0.5**2 + 0.5**2) 
        c = l / np.sqrt(2.0)
        
        # In Eq 14, Mz = sqrt(2)*(Ty1 - Ty3)[cite: 2]. 
        # Since T_yi = T_i * sin(beta) and T_zi = T_i * cos(beta), T_yi = T_zi * tan(beta)[cite: 2]
        k = np.sqrt(2.0) * np.tan(self.tilt_angle_rad)

        # Construction of the mapping matrix A: Wrench = A * [T_z1, T_z2, T_z3, T_z4]^T
        # Rows correspond directly to F_z, M_x, M_y, M_z[cite: 2]
        A = np.array([
            [ 1.0,  1.0,  1.0,  1.0], 
            [-c,    c,    c,   -c  ], 
            [ c,   -c,    c,   -c  ], 
            [ k,    0.0, -k,    0.0]  
        ])
        
        # Invert the matrix to solve for the required vertical thrust components (T_zi)
        A_inv = np.linalg.pinv(A)
        
        Wrench = np.array([total_z_thrust, torque_roll, torque_pitch, torque_yaw])
        T_z = np.dot(A_inv, Wrench)
        
        # Eq 13 specifies T_zi = T_i * cos(beta), so total motor thrust T_i = T_zi / cos(beta)[cite: 2]
        T_1 = T_z[0] / self.cos_g
        T_2 = T_z[1] / self.cos_g
        T_3 = T_z[2] / self.cos_g
        T_4 = T_z[3] / self.cos_g

        # Map the solved paper indices directly to the PyBullet links
        follower_thrust_cmds = {
            'iris_4': float(max(0.0, T_1)),
            'iris_1': float(max(0.0, T_2)),
            'iris_3': float(max(0.0, T_3)),
            'iris_2': float(max(0.0, T_4))
        }
        
        # All yaw generation is physically handled by the tilt vectors (Eq 14)[cite: 2]
        follower_torque_cmds = {name: [0.0, 0.0, 0.0] for name in self.r_arms.keys()}
            
        dummy_wrench = [total_z_thrust, torque_roll, torque_pitch, torque_yaw]
        
        return follower_thrust_cmds, follower_torque_cmds, dummy_wrench