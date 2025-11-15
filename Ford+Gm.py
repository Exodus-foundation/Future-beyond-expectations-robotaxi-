import math
import time

# --- Overview: The Hierarchical Planning Stack (Inspired by GM Cruise and former Ford Argo AI) ---
# This simulation models the core hierarchy used in many SAE Level 4 autonomous vehicles,
# like those developed by GM (via Cruise) and previously by Ford (via Argo AI).
# It consists of three main layers: Mission, Behavior, and Local Planning.
# The simulation assumes operation within a defined Operational Design Domain (ODD).

# --- 1. Data Structures ---

class VehicleState:
    """Represents the current state of the robotaxi (Ego Vehicle)."""
    def __init__(self, x, y, heading, speed):
        # Current position (meters)
        self.x = x
        self.y = y
        # Current heading (radians)
        self.heading = heading
        # Current speed (m/s)
        self.speed = speed

    def __repr__(self):
        return f"Pos: ({self.x:.2f}, {self.y:.2f}), Speed: {self.speed:.1f} m/s, Heading: {math.degrees(self.heading):.1f}°"

class TrafficObject:
    """Represents a dynamic or static object detected by the Perception system."""
    def __init__(self, obj_id, obj_type, x, y, speed=0.0, prediction=None):
        self.id = obj_id
        self.type = obj_type  # e.g., 'pedestrian', 'car', 'traffic_light'
        self.x = x
        self.y = y
        self.speed = speed
        self.prediction = prediction or [] # Predicted future path (x, y, t)

# --- 2. Mission Planner (Global Route) ---

class MissionPlanner:
    """
    Simulates the Global Planning layer. 
    In a real L4 system, this would use HD Maps and algorithms like A* or Dijkstra's to find the optimal high-level route.
    """
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        # Simplified global route: a list of key waypoints (x, y)
        self.global_waypoints = [
            (start.x, start.y),
            (100, 50),
            (150, 50), # Intersection approach
            (150, 100),
            (200, 150),
            (goal[0], goal[1])
        ]
        self.current_waypoint_index = 0
        print(f"Mission loaded: from {start.x, start.y} to {goal[0], goal[1]}.")

    def get_next_target_waypoint(self):
        """Returns the next major waypoint on the route."""
        if self.current_waypoint_index < len(self.global_waypoints):
            return self.global_waypoints[self.current_waypoint_index]
        return None

    def advance_waypoint(self):
        """Moves to the next waypoint."""
        self.current_waypoint_index += 1
        if self.current_waypoint_index >= len(self.global_waypoints):
            return False # Mission complete
        return True

# --- 3. Behavior Planner (FSM - Decision Making) ---

class BehaviorPlanner:
    """
    Simulates the Behavioral layer. Uses a Finite State Machine (FSM) 
    to decide the immediate high-level maneuver based on the current state 
    and perceived environment.
    """
    # Define FSM States
    STATE_CRUISE = "CRUISE"
    STATE_APPROACH_INTERSECTION = "APPROACH_INTERSECTION"
    STATE_STOP_AT_LIGHT = "STOP_AT_LIGHT"
    STATE_YIELD_PEDESTRIAN = "YIELD_PEDESTRIAN"
    STATE_MISSION_COMPLETE = "MISSION_COMPLETE"

    def __init__(self):
        self.state = self.STATE_CRUISE

    def get_next_behavior(self, ego_state, traffic_objects, next_waypoint):
        """
        Determines the next required action (behavior).
        """
        if next_waypoint is None:
            self.state = self.STATE_MISSION_COMPLETE
            return "TERMINATE"
        
        # Check for immediate safety/traffic constraints (Highest priority)
        for obj in traffic_objects:
            distance = math.sqrt((obj.x - ego_state.x)**2 + (obj.y - ego_state.y)**2)
            
            if obj.type == 'pedestrian' and distance < 15:
                # If a pedestrian is close, yield regardless of current state
                self.state = self.STATE_YIELD_PEDESTRIAN
                return "DECEL_TO_ZERO_FOR_YIELD"

            if obj.type == 'traffic_light' and distance < 30 and obj.prediction['color'] == 'RED':
                # Approaching a red light
                self.state = self.STATE_STOP_AT_LIGHT
                return "DECEL_TO_STOP_AT_MARKER"

        # Default maneuver: CRUISE towards the next waypoint
        if self.state != self.STATE_MISSION_COMPLETE:
            self.state = self.STATE_CRUISE
            return "MAINTAIN_SPEED_AND_FOLLOW_PATH"
        
        return "TERMINATE"


# --- 4. Local Planner (Trajectory Generation) ---

class LocalPlanner:
    """
    Simulates the Local Planning layer. Takes the high-level behavior 
    and generates a detailed, smooth, and safe trajectory (a series of 
    discrete target points) for the vehicle to follow.
    """
    def generate_trajectory(self, ego_state, next_waypoint, required_behavior):
        """Generates a list of (x, y, target_speed) points for the next 2 seconds."""
        trajectory = []
        num_steps = 20 # 10 steps per second (0.1s interval)
        time_step = 0.1
        
        print(f"   [Local] Behavior: {required_behavior}")

        if required_behavior == "TERMINATE":
            return [(ego_state.x, ego_state.y, 0.0)]
        
        # Determine target speed based on behavior
        target_speed = 10.0 # Default cruising speed (m/s)
        
        if required_behavior == "DECEL_TO_ZERO_FOR_YIELD":
            target_speed = 0.0
            
        elif required_behavior == "DECEL_TO_STOP_AT_MARKER":
            # Decelerate based on distance to the stop line/marker
            target_speed = max(0.0, ego_state.speed - 2.0) # Simple deceleration
            
        # Calculate ideal movement vector (towards next waypoint)
        dx = next_waypoint[0] - ego_state.x
        dy = next_waypoint[1] - ego_state.y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        target_heading = math.atan2(dy, dx)
        
        current_x = ego_state.x
        current_y = ego_state.y
        current_speed = ego_state.speed
        
        # Generate points
        for i in range(1, num_steps + 1):
            # Simple linear interpolation for demonstration:
            # In reality, this uses polynomial curves (splines) and trajectory optimization
            
            # Simple speed adjustment (P-controller logic)
            speed_command = current_speed + (target_speed - current_speed) * 0.1 
            speed_command = max(0.0, min(15.0, speed_command))

            # Move in the target direction
            new_x = current_x + speed_command * time_step * math.cos(target_heading)
            new_y = current_y + speed_command * time_step * math.sin(target_heading)

            trajectory.append((new_x, new_y, speed_command))
            
            # Update current state for the next step's calculation
            current_x, current_y = new_x, new_y
            current_speed = speed_command
            
        return trajectory


# --- 5. Main Simulation Loop ---

class RobotaxiSim:
    """The central loop that mimics the vehicle's execution process."""
    def __init__(self, start_pos, end_pos):
        self.ego_state = VehicleState(start_pos[0], start_pos[1], math.pi/4, 0.0)
        self.mission_planner = MissionPlanner(self.ego_state, end_pos)
        self.behavior_planner = BehaviorPlanner()
        self.local_planner = LocalPlanner()
        self.is_running = True
        
        # Mock up environment for L4 decision making (Perception output)
        self.environment = [
            TrafficObject(101, 'pedestrian', 140, 60, prediction=[(150, 70, 2.0)]),
            TrafficObject(201, 'traffic_light', 160, 50, prediction={'color': 'RED'}),
            TrafficObject(301, 'car', 10, 10, speed=5.0),
        ]

    def _execute_trajectory(self, trajectory):
        """Simulates the Control layer executing the first point of the trajectory."""
        if not trajectory:
            return

        # Use the first point of the trajectory as the immediate command
        next_x, next_y, next_speed = trajectory[0]

        # Calculate new heading
        dx = next_x - self.ego_state.x
        dy = next_y - self.ego_state.y

        # Only update if there is movement
        if abs(dx) > 0.01 or abs(dy) > 0.01:
            new_heading = math.atan2(dy, dx)
        else:
            new_heading = self.ego_state.heading

        # Update the vehicle state based on the executed command
        self.ego_state.x = next_x
        self.ego_state.y = next_y
        self.ego_state.speed = next_speed
        self.ego_state.heading = new_heading
        
    def run_cycle(self):
        """The main autonomous driving cycle (runs at 10Hz or faster)."""
        print("-" * 50)
        print(f"Cycle Start | Current State: {self.ego_state}")
        
        # 1. Perception/Localization (Simulated by environment list and ego_state)
        # In a real system, sensor fusion updates the environment and ego_state.
        
        # 2. Mission Planning (Global Goal)
        next_wp = self.mission_planner.get_next_target_waypoint()
        if next_wp is None:
            self.is_running = False
            print("Mission Complete! Robotaxi has arrived.")
            return

        # 3. Behavior Planning (Decision)
        behavior = self.behavior_planner.get_next_behavior(self.ego_state, self.environment, next_wp)

        # 4. Local Planning (Trajectory Generation)
        trajectory = self.local_planner.generate_trajectory(self.ego_state, next_wp, behavior)
        
        # 5. Control (Execution)
        self._execute_trajectory(trajectory)

        # Check if the vehicle has reached the immediate waypoint
        dist_to_wp = math.sqrt((next_wp[0] - self.ego_state.x)**2 + (next_wp[1] - self.ego_state.y)**2)
        if dist_to_wp < 5.0: # Tolerance of 5 meters
            self.mission_planner.advance_waypoint()
            print(f"*** Advanced to next waypoint! Distance remaining: {dist_to_wp:.2f} m ***")

        if self.ego_state.speed == 0.0 and behavior == "TERMINATE":
             self.is_running = False

        print(f"Cycle End | New State: {self.ego_state}")


# --- Execution ---
if __name__ == "__main__":
    
    # Define the high-level mission (simulated coordinates)
    START_POINT = (5, 5) 
    END_POINT = (200, 200)

    # Instantiate the Robotaxi and run the simulation
    robotaxi = RobotaxiSim(START_POINT, END_POINT)
    
    num_cycles = 0
    while robotaxi.is_running and num_cycles < 30: # Limit cycles for demonstration
        robotaxi.run_cycle()
        num_cycles += 1
        time.sleep(0.05) # Simulate a fast loop (20 Hz)
        
    print("\n--- Simulation Summary ---")
    print(f"Final State: {robotaxi.ego_state}")
    if not robotaxi.is_running and robotaxi.behavior_planner.state == robotaxi.behavior_planner.STATE_MISSION_COMPLETE:
        print("SAE L4 Mission successfully terminated within Operational Design Domain.")
    elif robotaxi.is_running:
        print("Simulation stopped after 30 cycles.")
