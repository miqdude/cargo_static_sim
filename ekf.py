import numpy as np

class EKFEstimator:
    def __init__(self):
        # State: [c_x, c_y]^T
        self.x = np.zeros((2, 1)) 
        
        # P: State Covariance (Initial uncertainty)
        self.P = np.eye(2) * 1.0  
        
        # Q: Process Noise Covariance
        # Extremely low because we assume the static payload doesn't physically move mid-flight
        self.Q = np.eye(2) * 1e-6 
        
        # R: Measurement Noise Covariance
        # Higher values reflect noisy angular acceleration readings from the IMU
        self.R = np.eye(2) * 0.1  

    def update(self, F_total, tau_geom, omega, angular_accel, inertia):
        """
        Runs one step of the EKF to predict and update the CoG estimate.
        """
        Ixx, Iyy, Izz = inertia
        wx, wy, wz = omega[0], omega[1], omega[2]
        tx, ty = tau_geom[0], tau_geom[1]
        
        # Measured angular acceleration
        alpha_meas = np.array([
            [angular_accel[0]], 
            [angular_accel[1]]
        ])

        # Prevent singularities
        if F_total < 1.0:
            return self.x.flatten()

        # ---------------------------------------------------------
        # 1. PREDICT STEP
        # ---------------------------------------------------------
        # The payload is static, so the prediction is simply the previous state
        x_pred = self.x
        P_pred = self.P + self.Q

        cx_pred = x_pred[0, 0]
        cy_pred = x_pred[1, 0]

        # ---------------------------------------------------------
        # 2. MEASUREMENT MODEL h(x)
        # ---------------------------------------------------------
        # Calculating expected angular acceleration based on current CoG guess
        h_x = np.array([
            [(tx - cy_pred * F_total - (wy * wz * (Izz - Iyy))) / Ixx],
            [(ty + cx_pred * F_total - (wx * wz * (Ixx - Izz))) / Iyy]
        ])

        # ---------------------------------------------------------
        # 3. JACOBIAN MATRIX (H)
        # ---------------------------------------------------------
        # Partial derivatives of h(x) with respect to c_x and c_y
        H = np.array([
            [0.0,           -F_total / Ixx],
            [F_total / Iyy, 0.0           ]
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
        self.P = (np.eye(2) - K @ H) @ P_pred

        return self.x.flatten() # Returns [c_x, c_y]