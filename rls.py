#!/usr/bin/env python3
import numpy as np

class RLSEstimator:
    def __init__(self, lambda_forgetting=0.995, initial_mass=1.0):
        """
        lambda_forgetting: Controls how quickly the filter forgets old data. 
        initial_mass: Warm start for the mass estimate to speed up convergence.
        """
        self.lam = lambda_forgetting
        
        # State: [c_x, c_y, mass]^T
        self.theta = np.zeros((3, 1))
        self.theta[2, 0] = initial_mass
        
        # Covariance Matrix (3x3)
        # Initialized high for c_x and c_y, but you can lower the 3rd diagonal 
        # if you are confident in your initial_mass guess.
        self.P = np.eye(3) * 1000.0 

    def update(self, F_total, tau_geom, angular_accel, linear_accel_z, roll, pitch, inertia, gravity=9.81):
        """
        Runs one step of the 3-State RLS algorithm based on telemetry.
        """
        Ixx, Iyy, _ = inertia
        tau_x_geom, tau_y_geom = tau_geom[0], tau_geom[1]
        alpha_x, alpha_y = angular_accel[0], angular_accel[1]

        # Prevent singularities if thrust is near zero (e.g., freefall/grounded)
        if F_total < 1.0:
            return self.theta.flatten()

        # Calculate Z-axis thrust projection (R33)
        R33 = np.cos(roll) * np.cos(pitch)

        # 1. Define the target measurement vector (y) - Shape: (3, 1)
        y = np.array([
            [Iyy * alpha_y - tau_y_geom],
            [tau_x_geom - Ixx * alpha_x],
            [F_total * R33]
        ])

        # 2. Define the observation matrix (phi) - Shape: (3, 3)
        phi = np.array([
            [F_total, 0.0, 0.0],
            [0.0, F_total, 0.0],
            [0.0, 0.0, linear_accel_z + gravity]
        ])

        # 3. Standard RLS Update Equations
        # S = lambda * I + phi @ P @ phi^T
        S = self.lam * np.eye(3) + phi @ self.P @ phi.T
        
        # Gain K = P @ phi^T @ S^-1
        K = self.P @ phi.T @ np.linalg.inv(S)

        # Update Parameter Estimate
        self.theta = self.theta + K @ (y - phi @ self.theta)
        
        # Update Covariance
        self.P = (self.P - K @ phi @ self.P) / self.lam

        # Returns [c_x, c_y, mass]
        return self.theta.flatten()