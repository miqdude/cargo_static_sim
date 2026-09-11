import numpy as np

class WaypointManager:
    def __init__(self, waypoints, acceptance_radius=0.2, max_speed=1.5):
        self.waypoints = waypoints
        self.current_wp_index = 0
        self.acceptance_radius = acceptance_radius
        self.max_speed = max_speed
        
        # The "virtual carrot" the drone actually follows
        self.virtual_pos = None

    def get_target_state(self, current_pos, dt):
        # 1. If all waypoints are reached, hover at the final one
        if self.current_wp_index >= len(self.waypoints):
            final_wp = np.array(self.waypoints[-1])
            return final_wp, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        target_wp = np.array(self.waypoints[self.current_wp_index])

        # 2. Check if the drone physically reached the current waypoint
        dist_to_wp = np.linalg.norm(current_pos - target_wp)
        if dist_to_wp < self.acceptance_radius:
            print(f"Reached Waypoint {self.current_wp_index}! Moving to next.")
            self.current_wp_index += 1
            return self.get_target_state(current_pos, dt) # Recurse to grab next point

        # 3. Initialize the virtual carrot at the drone's spawn position
        if self.virtual_pos is None:
            self.virtual_pos = np.copy(current_pos)

        # 4. Move the virtual carrot towards the waypoint at max_speed
        direction = target_wp - self.virtual_pos
        dist_to_target = np.linalg.norm(direction)

        if dist_to_target > 0:
            direction /= dist_to_target # Normalize the vector

        step_size = self.max_speed * dt
        
        if step_size > dist_to_target:
            # Snap to waypoint if the step overshoots it
            self.virtual_pos = target_wp
            vel = direction * (dist_to_target / dt)
        else:
            # Move the carrot forward
            self.virtual_pos += direction * step_size
            vel = direction * self.max_speed

        acc = np.zeros(3) # Can be derived from vel if needed, but 0 is usually fine here
        rpy = np.zeros(3)
        ang_vel = np.zeros(3)

        return self.virtual_pos, vel, acc, rpy, ang_vel