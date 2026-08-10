import os
import numpy as np

from quadrotor import Quadrotor

class PayloadEnv(object):
    def __init__(self, urdf_path, virtual_leader_urf_path, leader, p, gravity_mss = 9.81, debug = False, drone_max_thrust = 25):
        self.debug = debug
        self.pybullet = p
        self.gravity_mss = gravity_mss
        self.drone_max_thrust = drone_max_thrust
        self.starting_position = [0, 0, 0.2]

        # Load the assembled model safely using absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, urdf_path)
        
        # Start at Z=1.05 to clear the 1.0m landing gear
        self.robot_id = self.pybullet.loadURDF(full_path, basePosition=self.starting_position)

        if self.debug :
            self.leader_marker_id = self.pybullet.loadURDF(
                virtual_leader_urf_path, 
                basePosition=[0.0, 0.0, 1.05], 
                useFixedBase=True
            )
        
        self.leader = leader

        # Get the total number of joints/links in your drone
        num_joints = p.getNumJoints(self.robot_id)
        
        print("\n--- PYBULLET LINK MAP ---")
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            print(f"Index: {i} | Joint Name: {joint_name} | Link Name: {link_name}")
        print("-------------------------\n")

        self.drones = []
        self.map_links()

        # Data Logging
        self.log_errors_3d = []
        self.log_swing_angles = []
        self.log_swing_energy = []

        self.log_velocity = []
        self.log_ang_velocity = []

        self.rng = np.random.default_rng()

    def reset(self):
        """Resets the drone and spawns a NEW random CoG Anomaly."""
        
        # 1. Clean up the old anomaly if it exists
        if hasattr(self, 'cog_constraint'):
            self.pybullet.removeConstraint(self.cog_constraint)
        if hasattr(self, 'shifted_mass_id'):
            self.pybullet.removeBody(self.shifted_mass_id)

        # 2. Reset the main drone back to the starting hover position
        start_pos = self.starting_position
        start_orientation = self.pybullet.getQuaternionFromEuler([0, 0, 0])
        self.pybullet.resetBasePositionAndOrientation(self.robot_id, start_pos, start_orientation)
        self.pybullet.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])

        # 3. Generate a Random CoG Shift (between -15cm and +15cm)
        # We only shift X and Y. Z stays slightly elevated to sit on the frame.
        # x_shift = np.random.uniform(-0.2, 0.2)
        # y_shift = np.random.uniform(-0.2, 0.2)
        x_shift = 0.15
        y_shift = 0.15
        self.payload_offset = [x_shift, y_shift, 0.05]

        random_payload_mass = self.rng.uniform(0.5,1)

        # 4. Re-spawn the red box
        cog_shift_mass = 1.0 # 1 kg
        box_half_extents = [0.05, 0.05, 0.05]
        
        visual_shape_id = self.pybullet.createVisualShape(
            shapeType=self.pybullet.GEOM_BOX,
            halfExtents=box_half_extents,
            rgbaColor=[1, 0, 0, 1] 
        )
        collision_shape_id = self.pybullet.createCollisionShape(
            shapeType=self.pybullet.GEOM_BOX,
            halfExtents=box_half_extents
        )

        spawn_pos = [start_pos[0] + x_shift, start_pos[1] + y_shift, start_pos[2]]
        
        self.shifted_mass_id = self.pybullet.createMultiBody(
            baseMass=cog_shift_mass,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=spawn_pos
        )

        self.pybullet.setCollisionFilterPair(
            self.robot_id, self.shifted_mass_id, -1, -1, enableCollision=0
        )

        # 5. Weld it to the frame and save the constraint ID so we can delete it later
        self.cog_constraint = self.pybullet.createConstraint(
            parentBodyUniqueId=self.robot_id,   
            parentLinkIndex=-1,                 
            childBodyUniqueId=self.shifted_mass_id,
            childLinkIndex=-1,
            jointType=self.pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=self.payload_offset,
            childFramePosition=[0, 0, 0]
        )
        
        # Clear the logs for the new episode
        self.log_errors_3d = []
        
        # Return the true offset so the logger can record what the PINN is supposed to guess!
        return self.payload_offset, random_payload_mass

    def map_links(self):
        motor_map = {}
        base_map = {}
        
        self.pendulum_roll_idx = 0
        self.pendulum_pitch_idx = 1

        num_joints = self.pybullet.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = self.pybullet.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            joint_name = joint_info[1].decode('utf-8')

            if "rotor" in link_name:
                drone_name, rotor_name = link_name.split('/')
                if drone_name not in motor_map:
                    motor_map[drone_name] = {}
                motor_map[drone_name][rotor_name] = i
                
            elif "base_link" in link_name and "iris" in link_name:
                drone_name = link_name.split('/')[0]
                base_map[drone_name] = i

        # Instantiate the 4 Quadrotor objects
        for drone_name in ['iris_1', 'iris_2', 'iris_3', 'iris_4']:
            indices = [
                motor_map[drone_name]['rotor_0'],
                motor_map[drone_name]['rotor_1'],
                motor_map[drone_name]['rotor_2'],
                motor_map[drone_name]['rotor_3']
            ]
            drone_obj = Quadrotor(
                self.pybullet,
                self.robot_id,
                drone_name,
                base_map[drone_name], 
                indices,
                self.drone_max_thrust)
            self.drones.append(drone_obj)

    def step(self,
            m_frame: float,
            m_payload: float,
            dt: float,
            time_now: float,
            follower_thrust_cmds,
            follower_torque_cmds,
        ):

        # 1. Read Global Frame State for the Leader
        pos, quat = self.pybullet.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = self.pybullet.getBaseVelocity(self.robot_id)
        current_rpy = self.pybullet.getEulerFromQuaternion(quat)

        # Update all 4 drones
        for drone in self.drones:
            # The command is no longer a [vx, vy, vz] vector. It is a single float (Newtons)
            cmd_thrust = follower_thrust_cmds[drone.name]
            cmd_torque = follower_torque_cmds[drone.name]

            # Apply the physics
            drone.update(total_thrust_cmd=cmd_thrust, torque_cmd=cmd_torque)
             
        self.pybullet.stepSimulation()
        
        # Extract main frame (root link) state
        pos, quat = self.pybullet.getBasePositionAndOrientation(self.robot_id)
        vel, ang_vel = self.pybullet.getBaseVelocity(self.robot_id)
        current_rpy = self.pybullet.getEulerFromQuaternion(quat)

        # log the data to get the acceleration
        # in the above step
        self.log_velocity.append(vel)
        self.log_ang_velocity.append(ang_vel)
        
        return pos, current_rpy, vel, ang_vel

    def get_logging(self):
        return self.log_errors_3d, self.log_swing_angles, self.log_swing_energy