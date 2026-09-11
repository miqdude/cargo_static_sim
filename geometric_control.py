import numpy as np

class GeometricControl:
    def __init__(self, mass_total=7.8):
        # System Properties
        self.M_total = mass_total  # kg
        self.g = 9.81
        self.dt = 1.0 / 240.0
        
        # Approximate Inertia Tensor (J) for the rigid frame
        # You may need to tune these values based on your PyBullet URDF
        self.J = np.diag([0.5, 0.5, 0.8]) 

        # 1. Position Gains (kx, kv)
        self.kx = np.array([5.0, 5.0, 8.0])
        self.kv = np.array([3.0, 3.0, 4.0])
        self.ki = np.array([0.1, 0.1, 0.5]) # Optional integral term for steady-state error
        self.integral_pos_error = np.zeros(3)

        # 2. Attitude Gains (kR, kOmega)
        # Geometric gains are typically much higher than PID gains
        self.kR = np.array([15.0, 15.0, 5.0])
        self.kOmega = np.array([4.0, 4.0, 2.0])

        # 3. Geometric Mounting Positions (Same as before)
        self.r_arms = {
            'iris_1': np.array([ 0.5,  0.5, 0.05]), 
            'iris_2': np.array([ 0.5, -0.5, 0.05]), 
            'iris_3': np.array([-0.5,  0.5, 0.05]), 
            'iris_4': np.array([-0.5, -0.5, 0.05])  
        }
        self.c_yaw = [1.0, -1.0, -1.0, 1.0]

    def _euler_to_rotation_matrix(self, rpy):
        """ Converts Roll, Pitch, Yaw to a 3x3 Rotation Matrix (Z-Y-X convention) """
        phi, theta, psi = rpy
        
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi), np.cos(phi)]])
        
        R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
        
        R_z = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
                        
        return R_z @ R_y @ R_x

    def _vee_map(self, R):
        """ Extracts the vector from a skew-symmetric matrix """
        return np.array([R[2, 1], R[0, 2], R[1, 0]])

    def get_allocation_matrix(self, cog_x=0.0, cog_y=0.0):
        A = np.zeros((4, 4))
        A[0, :] = 1.0
        for i, (name, pos) in enumerate(self.r_arms.items()):
            eff_x = pos[0] - cog_x
            eff_y = pos[1] - cog_y
            A[1, i] = eff_y           # Roll Moment (Y arm)
            A[2, i] = -eff_x          # Pitch Moment (-X arm)
            A[3, i] = self.c_yaw[i]   # Yaw Moment
        return A

    def _calculate_total_cog(self, m_frame, m_payload, payload_offset):
        m_tot = m_frame + m_payload
        if m_tot <= 0.0: return 0.0, 0.0
        cog_x = (m_payload * payload_offset[0]) / m_tot
        cog_y = (m_payload * payload_offset[1]) / m_tot
        return cog_x, cog_y

    def get_follower_commands(
            self,
            payload_offset,
            dt,
            current_pos, current_rpy,
            current_vel, current_ang_vel,
            target_pos,
            target_vel, target_ang_vel, 
            target_acc, target_yaw,
            m_frame, m_payload):

        # ---------------------------------------------------------
        # 1. TRANSLATIONAL CONTROL (Position to Force)
        # ---------------------------------------------------------
        e_p = np.array(current_pos) - np.array(target_pos)
        e_v = np.array(current_vel) - np.array(target_vel)
        
        self.integral_pos_error += e_p * dt
        self.integral_pos_error = np.clip(self.integral_pos_error, -5.0, 5.0)

        # Desired Force Vector (F_des)
        # F = -kx*ep - kv*ev - ki*ei + m*g*e3 + m*ad
        e3 = np.array([0.0, 0.0, 1.0])
        F_des = (-self.kx * e_p) - (self.kv * e_v) - (self.ki * self.integral_pos_error) \
                + (self.M_total * self.g * e3) + (self.M_total * np.array(target_acc))

        # Current Rotation Matrix
        R_current = self._euler_to_rotation_matrix(current_rpy)

        # Total Thrust Command (Projection of F_des onto current body Z-axis)
        b3_current = R_current[:, 2]
        total_thrust = np.dot(F_des, b3_current)
        total_thrust = max(0.0, total_thrust) # Prevent negative thrust

        # ---------------------------------------------------------
        # 2. ROTATIONAL CONTROL (Force to Desired Attitude)
        # ---------------------------------------------------------
        # Desired Body Z-axis (b3_des) aligns with F_des
        norm_F = np.linalg.norm(F_des)
        if norm_F > 0:
            b3_des = F_des / norm_F
        else:
            b3_des = e3

        # Desired Yaw Direction (b1_c)
        yaw_cmd = target_yaw[2] if isinstance(target_yaw, (list, tuple, np.ndarray)) else target_yaw
        b1_c = np.array([np.cos(yaw_cmd), np.sin(yaw_cmd), 0.0])

        # Desired Body Y-axis (b2_des) = b3_des x b1_c
        b2_des = np.cross(b3_des, b1_c)
        norm_b2 = np.linalg.norm(b2_des)
        if norm_b2 > 0:
            b2_des = b2_des / norm_b2
        else:
            b2_des = np.array([0.0, 1.0, 0.0])

        # Desired Body X-axis (b1_des) = b2_des x b3_des
        b1_des = np.cross(b2_des, b3_des)

        # Construct Desired Rotation Matrix
        R_des = np.column_stack((b1_des, b2_des, b3_des))

        # ---------------------------------------------------------
        # 3. ATTITUDE ERROR & TORQUE COMMAND
        # ---------------------------------------------------------
        # Attitude Error on SO(3 manifold: eR = 0.5 * vee(Rd^T R - R^T Rd)
        error_matrix = 0.5 * (R_des.T @ R_current - R_current.T @ R_des)
        e_R = self._vee_map(error_matrix)

        # Angular Velocity Error (Assuming target_ang_vel is small/zero for hover)
        e_omega = np.array(current_ang_vel) - np.array(target_ang_vel)

        # Torque Command using Geometric Control Law
        # Tau = -kR*eR - kOmega*eOmega + (omega x J*omega)
        omega = np.array(current_ang_vel)
        gyro_term = np.cross(omega, self.J @ omega)
        
        tau_des = (-self.kR * e_R) - (self.kOmega * e_omega) + gyro_term

        # ---------------------------------------------------------
        # 4. ALLOCATION & COMMAND PACKAGING
        # ---------------------------------------------------------
        Wrench = [total_thrust, tau_des[0], tau_des[1], tau_des[2]]

        guess_cog_x, guess_cog_y = self._calculate_total_cog(m_frame, m_payload, payload_offset)
        
        A_final = self.get_allocation_matrix(guess_cog_x, guess_cog_y)
        A_inv = np.linalg.pinv(A_final)
        quad_thrusts = np.dot(A_inv, Wrench)

        follower_thrust_cmds = {}
        follower_torque_cmds = {}
        yaw_torque_per_drone = tau_des[2] / 4.0
        
        for i, name in enumerate(self.r_arms.keys()):
            follower_thrust_cmds[name] = float(max(0.0, quad_thrusts[i]))
            follower_torque_cmds[name] = [0.0, 0.0, float(yaw_torque_per_drone)]
            
        return follower_thrust_cmds, follower_torque_cmds, Wrench