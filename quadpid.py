import numpy as np

class QuadPIDController:
    def __init__(self, KP, KI, KD):
        self.integral_error = np.zeros(3)
        self.KP = KP
        self.KI = KI
        self.KD = KD
        
    def compute(self, current_rpy, target_rpy, current_ang_vel, dt):
        """Calculates torque commands based on orientation error for a single drone."""
        error = np.array(target_rpy) - np.array(current_rpy)
        error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi # Wrap yaw

        p_term = self.KP * error
        self.integral_error += error * dt
        i_term = self.KI * self.integral_error
        
        # Derivative on measurement (gyro) prevents "kicks" when the setpoint changes
        d_term = self.KD * (0.0 - current_ang_vel) 

        return p_term + i_term + d_term
