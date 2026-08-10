import numpy as np

class RLSEstimator:
    def __init__(self, lambda_forgetting=0.995):
        """
        lambda_forgetting: Controls how quickly the filter forgets old data. 
        0.99 to 0.999 is standard for static/slow-moving parameters.
        """
        self.lam = lambda_forgetting
        
        # State: [c_x, c_y]^T
        self.theta = np.zeros((2, 1)) 
        
        # Covariance Matrix (initialized very high to indicate low initial confidence)
        self.P = np.eye(2) * 1000.0 

    def update(self, F_total, tau_geom, angular_accel, inertia):
        """
        Runs one step of the RLS algorithm based on telemetry.
        """
        Ixx, Iyy, _ = inertia
        tau_x_geom, tau_y_geom = tau_geom[0], tau_geom[1]
        alpha_x, alpha_y = angular_accel[0], angular_accel[1]

        # 1. Define the target measurement matrix (y)
        # Derived from: c_x * F = I_yy * alpha_y - tau_y_geom
        # Derived from: c_y * F = tau_x_geom - I_xx * alpha_x
        y = np.array([
            [Iyy * alpha_y - tau_y_geom],
            [tau_x_geom - Ixx * alpha_x]
        ])

        # 2. Define the observation matrix (phi)
        phi = np.array([
            [F_total, 0.0],
            [0.0, F_total]
        ])

        # 3. Prevent singularities if thrust is near zero (e.g., freefall)
        if F_total < 1.0:
            return self.theta.flatten()

        # 4. Standard RLS Update Equations
        # S = lambda * I + phi @ P @ phi^T
        S = self.lam * np.eye(2) + phi @ self.P @ phi.T
        
        # Gain K = P @ phi^T @ S^-1
        K = self.P @ phi.T @ np.linalg.inv(S)

        # Update Parameter Estimate
        self.theta = self.theta + K @ (y - phi @ self.theta)
        
        # Update Covariance
        self.P = (self.P - K @ phi @ self.P) / self.lam

        return self.theta.flatten() # Returns [c_x, c_y]