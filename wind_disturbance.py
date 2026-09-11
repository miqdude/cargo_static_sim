def step(self, ...):
        # ... (Your existing control loop and motor force application code) ...

        # =========================================================
        # ENVIRONMENTAL DISTURBANCE: AERODYNAMIC DRAG / WIND
        # =========================================================
        # 1. Base Directional Wind (e.g., constant push along X and Y)
        wind_base = np.array([4.0, 2.5, 0.0]) 

        # 2. Time-Varying Gusts (Sinusoidal sweeps to simulate swirling air)
        # Assuming you have a variable tracking simulation time, e.g., 'sim_time'
        wind_gust = np.array([
            np.sin(sim_time * 1.5) * 3.0,  # Swirling X gust
            np.cos(sim_time * 1.2) * 4.0,  # Swirling Y gust
            np.sin(sim_time * 2.5) * 1.0   # Mild vertical updrafts
        ])

        # 3. High-Frequency Turbulence (Gaussian noise)
        wind_turbulence = np.random.normal(loc=0.0, scale=0.8, size=3)

        # Calculate Total Aerodynamic Force (Newtons)
        total_wind_force = wind_base + wind_gust + wind_turbulence

        # 4. Apply the Disturbance to the Rigid Frame
        # NOTE: 'self.robot_id' must match your loaded URDF ID.
        # linkIndex=-1 targets the main chassis (center of the rigid frame).
        p.applyExternalForce(
            objectUniqueId=self.robot_id,  
            linkIndex=-1,                  
            forceObj=total_wind_force.tolist(),
            posObj=[0.1, 0.05, 0.0],       # Apply slightly off-center to induce aerodynamic torque!
            flags=p.WORLD_FRAME            
        )
        # =========================================================

        # Step the physics engine
        p.stepSimulation()
        
        # ... (Return state telemetry to the ZMQ socket) ...