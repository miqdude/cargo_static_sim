import numpy as np

class EKFEstimator:
    def __init__(self, initial_mass=6.8):
        # State: [c_x, c_y, mass]^T
        self.x = np.zeros((3, 1)) 
        self.x[2, 0] = initial_mass
        
        # P: State Covariance (Initial uncertainty)
        self.P = np.eye(3) * 1.0  
        
        # Q: Process Noise Covariance
        # Extremely low because we assume the static payload doesn't physically move or change mass mid-flight
        self.Q = np.eye(3) * 1e-6 
        
        # R: Measurement Noise Covariance
        # 3x3 matrix: higher values reflect noisy angular/linear acceleration readings from the IMU
        self.R = np.eye(3) * 0.1  

    def update(self, F_total, tau_geom, omega, angular_accel, linear_accel_z, roll, pitch, inertia, gravity=9.81):
        """
        Runs one step of the EKF to predict and update the CoG and Mass estimates.
        """
        Ixx, Iyy, Izz = inertia
        wx, wy, wz = omega[0], omega[1], omega[2]
        tx, ty = tau_geom[0], tau_geom[1]
        
        # Measured accelerations (Z-axis linear acceleration appended)
        alpha_meas = np.array([
            [angular_accel[0]], 
            [angular_accel[1]],
            [linear_accel_z]
        ])

        # Prevent singularities during freefall or when grounded
        if F_total < 1.0:
            return self.x.flatten()

        # Calculate Z-axis thrust projection
        R33 = np.cos(roll) * np.cos(pitch)

        # ---------------------------------------------------------
        # 1. PREDICT STEP
        # ---------------------------------------------------------
        # The payload is static, so the prediction is simply the previous state
        x_pred = self.x
        P_pred = self.P + self.Q

        cx_pred = x_pred[0, 0]
        cy_pred = x_pred[1, 0]
        
        # Prevent division by zero in the measurement model
        m_pred = max(x_pred[2, 0], 0.1) 

        # ---------------------------------------------------------
        # 2. MEASUREMENT MODEL h(x)
        # ---------------------------------------------------------
        # Calculating expected accelerations based on current CoG and Mass guesses
        h_x = np.array([
            [(tx - cy_pred * F_total - (wy * wz * (Izz - Iyy))) / Ixx],
            [(ty + cx_pred * F_total - (wx * wz * (Ixx - Izz))) / Iyy],
            [(F_total * R33) / m_pred - gravity]
        ])

        # ---------------------------------------------------------
        # 3. JACOBIAN MATRIX (H)
        # ---------------------------------------------------------
        # Partial derivatives of h(x) with respect to c_x, c_y, and mass
        H = np.array([
            [0.0,           -F_total / Ixx,  0.0],
            [F_total / Iyy,  0.0,            0.0],
            [0.0,            0.0,           -(F_total * R33) / (m_pred**2)]
        ])

        # ---------------------------------------------------------
        # 4. UPDATE STEP
        # ---------------------------------------------------------
        # Innovation (Error between measured and predicted)
        y_residual = alpha_meas - h_x 
        
        # Innovation Covariance (S)
        S = H @ P_pred @ H.T + self.R
        
        # Kalman Gain (K)
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Final State and Covariance Update
        self.x = x_pred + K @ y_residual
        self.P = (np.eye(3) - K @ H) @ P_pred

        return self.x.flatten() # Returns [c_x, c_y, mass]