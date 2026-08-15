import torch
import torch.nn as nn
import torch.nn.functional as F

class PINNInverseEstimator(nn.Module):
    def __init__(self):
        super(PINNInverseEstimator, self).__init__()
        
        # 1. The Trajectory Surrogate
        # Input: time (t) -> Output: The 12-DoF Flight State
        self.state_approximator = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 12) 
        )
        
        # 2. The Unknown Physical Parameters (Thesis Targets)
        # Initialized with baseline guesses. The optimizer will adjust these!
        self.mass = nn.Parameter(torch.tensor([1.0]))   
        self.cog_x = nn.Parameter(torch.tensor([0.0]))
        self.cog_y = nn.Parameter(torch.tensor([0.0]))

        # Also Gues the new inertia
        self.J_xx = nn.Parameter(torch.tensor([0.0]))   
        self.J_yy = nn.Parameter(torch.tensor([0.0]))   
        self.J_zz = nn.Parameter(torch.tensor([0.0]))   


    def forward(self, t):
        # Predict the continuous flight state at time 't'
        predicted_states = self.state_approximator(t)
        
        # Enforce positive mass physically using Softplus
        safe_mass = F.softplus(self.mass)
        J_xx = F.softplus(self.J_xx)
        J_yy = F.softplus(self.J_yy)
        J_zz = F.softplus(self.J_zz)
        
        return predicted_states, safe_mass, self.cog_x, self.cog_y, J_xx, J_yy, J_zz

def get_rotation_matrix(rpy):
    """ Converts a batch of Roll, Pitch, Yaw into 3x3 Rotation Matrices """
    cr, sr = torch.cos(rpy[:, 0]), torch.sin(rpy[:, 0])
    cp, sp = torch.cos(rpy[:, 1]), torch.sin(rpy[:, 1])
    cy, sy = torch.cos(rpy[:, 2]), torch.sin(rpy[:, 2])
    
    R = torch.zeros((rpy.shape[0], 3, 3), device=rpy.device)
    R[:, 0, 0] = cp * cy
    R[:, 0, 1] = sr * sp * cy - cr * sy
    R[:, 0, 2] = cr * sp * cy + sr * sy
    R[:, 1, 0] = cp * sy
    R[:, 1, 1] = sr * sp * sy + cr * cy
    R[:, 1, 2] = cr * sp * sy - sr * cy
    R[:, 2, 0] = -sp
    R[:, 2, 1] = sr * cp
    R[:, 2, 2] = cr * cp
    return R

def train_pinn_step(
    pinn_model, optimizer, 
    true_times, true_states, 
    total_thrust, applied_torques, 
    gravity=9.81, lambda_d=100.0, lambda_t=1.0, lambda_r=1.0
):
    """
    NOTE: J_base has been REMOVED from the arguments. 
    The network now discovers the Inertia natively.
    """
    optimizer.zero_grad()
    
    # =========================================================
    # 1. FORWARD PASS (Now extracting Inertia!)
    # =========================================================
    # Your PINN model must be updated to return 3 new parameters
    pred_states, pred_mass, pred_cog_x, pred_cog_y, pred_Jxx, pred_Jyy, pred_Jzz = pinn_model(true_times)
    
    loss_data = F.mse_loss(pred_states, true_states)
    
    pred_pos = pred_states[:, 0:3]
    pred_vel = pred_states[:, 3:6]
    pred_rpy = pred_states[:, 6:9]
    pred_omega = pred_states[:, 9:12]

    # =========================================================
    # 2. CONTINUOUS KINEMATICS (Autograd on Velocity)
    # =========================================================
    lin_acc_x = torch.autograd.grad(pred_vel[:, 0].sum(), true_times, create_graph=True)[0]
    lin_acc_y = torch.autograd.grad(pred_vel[:, 1].sum(), true_times, create_graph=True)[0]
    lin_acc_z = torch.autograd.grad(pred_vel[:, 2].sum(), true_times, create_graph=True)[0]
    lin_acc = torch.cat([lin_acc_x, lin_acc_y, lin_acc_z], dim=1) 
    
    ang_acc_x = torch.autograd.grad(pred_omega[:, 0].sum(), true_times, create_graph=True)[0]
    ang_acc_y = torch.autograd.grad(pred_omega[:, 1].sum(), true_times, create_graph=True)[0]
    ang_acc_z = torch.autograd.grad(pred_omega[:, 2].sum(), true_times, create_graph=True)[0]
    ang_acc = torch.cat([ang_acc_x, ang_acc_y, ang_acc_z], dim=1) 

    # =========================================================
    # 3. LAGRANGIAN SYSTEM ENERGIES (T and V)
    # =========================================================
    # T_trans = 0.5 * pred_mass * torch.sum(pred_vel**2, dim=1, keepdim=True)
    # V_potential = pred_mass * gravity * pred_pos[:, 2].unsqueeze(1)
    
    # # NEW: Construct the predicted Inertia vector for element-wise math
    # pred_J_diag = torch.cat([pred_Jxx, pred_Jyy, pred_Jzz], dim=1)
    
    # # Rotational Kinetic Energy: T_rot = 0.5 * w^T * J * w
    # # Because J is a diagonal matrix, J * w is just element-wise multiplication
    # J_omega = pred_J_diag * pred_omega 
    # T_rot = 0.5 * torch.sum(pred_omega * J_omega, dim=1, keepdim=True)

    mass_safe = pred_mass.view(-1, 1)
    
    # Translational Kinetic Energy
    T_trans = 0.5 * mass_safe * torch.sum(pred_vel**2, dim=1, keepdim=True)
    
    # Potential Energy is a SCALAR. We hardcode 9.81 to prevent 3D Vector crashes.
    V_potential = mass_safe * 9.81 * pred_pos[:, 2].unsqueeze(1)
    
    # Safely force all Inertia parameters into Column Vectors before concatenating
    pred_J_diag = torch.cat([
        pred_Jxx.view(-1, 1), 
        pred_Jyy.view(-1, 1), 
        pred_Jzz.view(-1, 1)
    ], dim=1)
    
    # Rotational Kinetic Energy: T_rot = 0.5 * w^T * J * w
    J_omega = pred_J_diag * pred_omega 
    T_rot = 0.5 * torch.sum(pred_omega * J_omega, dim=1, keepdim=True)

    # =========================================================
    # 4. TRANSLATIONAL RESIDUAL (Euler-Lagrange)
    # =========================================================
    dp_trans_dt = pred_mass * lin_acc 
    dV_dp = torch.autograd.grad(V_potential.sum(), pred_pos, create_graph=True)[0]

    R_matrix = get_rotation_matrix(pred_rpy)
    e3 = torch.tensor([0.0, 0.0, 1.0], device=true_times.device).view(1, 3, 1)
    local_thrust = total_thrust.unsqueeze(2) * e3 
    global_thrust = torch.bmm(R_matrix, local_thrust).squeeze(2) 
    
    res_trans = (dp_trans_dt + dV_dp) - global_thrust
    loss_trans = torch.mean(torch.sum(res_trans**2, dim=1))

    # =========================================================
    # 5. ROTATIONAL RESIDUAL (Euler-Poincaré)
    # =========================================================
    angular_momentum = torch.autograd.grad(T_rot.sum(), pred_omega, create_graph=True)[0]
    
    # dL/dt = J * ang_acc (Element-wise because J is diagonal)
    dL_dt = pred_J_diag * ang_acc
    
    gyro_torque = torch.cross(pred_omega, angular_momentum, dim=1)
    
    r_cog = torch.cat([
        pred_cog_x.expand(true_times.shape[0], 1), 
        pred_cog_y.expand(true_times.shape[0], 1), 
        torch.zeros((true_times.shape[0], 1), device=true_times.device)
    ], dim=1)
    
    force_vector = torch.zeros_like(r_cog)
    force_vector[:, 2] = total_thrust.squeeze(1)
    predicted_disturbance = torch.cross(r_cog, force_vector, dim=1)
    
    res_rot = (dL_dt + gyro_torque) - (applied_torques - predicted_disturbance)
    loss_rot = torch.mean(torch.sum(res_rot**2, dim=1))

    # =========================================================
    # 6. BACKPROPAGATION
    # =========================================================
    total_loss = (lambda_d * loss_data) + (lambda_t * loss_trans) + (lambda_r * loss_rot)
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return total_loss, loss_data, loss_trans, loss_rot

# def train_pinn_step(
#     pinn_model, optimizer, 
#     true_times, true_states, 
#     total_thrust, applied_torques, J_base, 
#     gravity=9.81, lambda_d=1.0, lambda_t=1.0, lambda_r=100.0
# ):
#     """
#     true_times: Tensor of shape (Batch, 1) -> MUST have requires_grad=True
#     true_states: Tensor of shape (Batch, 12) from PyBullet telemetry
#     """
#     optimizer.zero_grad()
    
#     # =========================================================
#     # 1. FORWARD PASS (Guess the trajectory and parameters)
#     # =========================================================
#     pred_states, pred_mass, pred_cog_x, pred_cog_y = pinn_model(true_times)
    
#     # =========================================================
#     # 2. DATA LOSS (Anchor the network to the PyBullet reality)
#     # =========================================================
#     # The network MUST accurately clone the flight path to take valid derivatives
#     loss_data = F.mse_loss(pred_states, true_states)
    
#     # Extract the NETWORK'S predicted kinematics for the physics loss
#     pred_vel = pred_states[:, 3:6]
#     pred_rpy = pred_states[:, 6:9]
#     pred_omega = pred_states[:, 9:12]

#     # =========================================================
#     # 3. CONTINUOUS DERIVATIVES (Autograd on the Network's output)
#     # =========================================================
#     # Linear Acceleration d(v)/dt
#     lin_acc_x = torch.autograd.grad(pred_vel[:, 0].sum(), true_times, create_graph=True)[0]
#     lin_acc_y = torch.autograd.grad(pred_vel[:, 1].sum(), true_times, create_graph=True)[0]
#     lin_acc_z = torch.autograd.grad(pred_vel[:, 2].sum(), true_times, create_graph=True)[0]
#     lin_acc = torch.cat([lin_acc_x, lin_acc_y, lin_acc_z], dim=1) 
    
#     # Angular Acceleration d(w)/dt
#     ang_acc_x = torch.autograd.grad(pred_omega[:, 0].sum(), true_times, create_graph=True)[0]
#     ang_acc_y = torch.autograd.grad(pred_omega[:, 1].sum(), true_times, create_graph=True)[0]
#     ang_acc_z = torch.autograd.grad(pred_omega[:, 2].sum(), true_times, create_graph=True)[0]
#     ang_acc = torch.cat([ang_acc_x, ang_acc_y, ang_acc_z], dim=1) 

#     # =========================================================
#     # 4. TRANSLATIONAL RESIDUAL (Optimizing Mass)
#     # =========================================================
#     g_vector = torch.tensor([0.0, 0.0, gravity], device=true_times.device).view(1, 3)
#     lhs_trans = pred_mass * (lin_acc + g_vector)
    
#     R_matrix = get_rotation_matrix(pred_rpy)
#     e3 = torch.tensor([0.0, 0.0, 1.0], device=true_times.device).view(1, 3, 1)
#     local_thrust = total_thrust.unsqueeze(2) * e3 
#     global_thrust = torch.bmm(R_matrix, local_thrust).squeeze(2) 
    
#     res_trans = lhs_trans - global_thrust
#     loss_trans = torch.mean(torch.sum(res_trans**2, dim=1))

#     # =========================================================
#     # 5. ROTATIONAL RESIDUAL (Optimizing CoG)
#     # =========================================================
#     inertial_torque = torch.matmul(ang_acc, J_base.T) 
#     J_omega = torch.matmul(pred_omega, J_base.T)
#     gyro_torque = torch.cross(pred_omega, J_omega, dim=1)
#     flight_torque = inertial_torque + gyro_torque
    
#     true_payload_torque = applied_torques - flight_torque
    
#     # Construct the CoG vector [x, y, 0]
#     r_cog = torch.cat([
#         pred_cog_x.expand(true_times.shape[0], 1), 
#         pred_cog_y.expand(true_times.shape[0], 1), 
#         torch.zeros((true_times.shape[0], 1), device=true_times.device)
#     ], dim=1)
    
#     force_vector = torch.zeros_like(r_cog)
#     force_vector[:, 2] = total_thrust.squeeze(1)
    
#     predicted_disturbance = torch.cross(r_cog, force_vector, dim=1)
    
#     res_rot = predicted_disturbance - true_payload_torque
#     loss_rot = torch.mean(torch.sum(res_rot**2, dim=1))

#     # =========================================================
#     # 6. BACKPROPAGATION
#     # =========================================================
#     total_loss = (lambda_d * loss_data) + (lambda_t * loss_trans) + (lambda_r * loss_rot)
    
#     total_loss.backward()
#     # Clip gradients to keep the rigid body math stable
#     torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), max_norm=1.0)
#     optimizer.step()
    
#     return total_loss, loss_data, loss_trans, loss_rot