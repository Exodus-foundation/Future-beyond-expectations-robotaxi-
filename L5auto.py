import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random

# --- Vehicle Constants ---
MAX_STEER_ANGLE = 0.52 # Radians (~30 degrees)
MAX_ACCELERATION = 4.0 # m/s^2
WHEELBASE = 2.8 # meters
MAX_SPEED = 30.0 # m/s (108 km/h)

# --- LLM API Setup (Mocked for constrained environment) ---
# NOTE: In a real system, this would be an async fetch call to the Gemini API.
# We are mocking the response structure here for stability in this environment.

def mock_llm_api_call(prompt: str) -> Dict:
    """Simulates a call to a cognitive planner (Gemini API)."""
    
    # Simple, deterministic mock responses based on key tasks
    if "Assess current safety profile" in prompt:
        # High confidence, no immediate threats
        return {
            "confidence_score": 0.95,
            "risk_summary": "Low intrinsic risk. External risks managed by Guarded Cruise.",
            "speed_adjustment_factor": 1.0, # Normal speed
            "lateral_shift_m": 0.0 # No shift
        }
    elif "Analyze THREAT_LEVEL_2" in prompt:
        # Security threat detected, suggest cautious behavior
        return {
            "confidence_score": 0.70,
            "risk_summary": "Active Level 2 anomaly detected. Prioritize central lane keeping and reduced speed.",
            "speed_adjustment_factor": 0.5, # Half speed
            "lateral_shift_m": 0.0 # Maintain center
        }
    elif "Suggest evasive maneuver" in prompt:
        # Evasion triggered, suggest a sharp shift
        return {
            "confidence_score": 0.85,
            "risk_summary": "Executing forced evasion. Target clear path right.",
            "speed_adjustment_factor": 1.0, # Maintain speed during evasion
            "lateral_shift_m": 2.0 # Sharp shift to the right
        }
    
    # Default safe fallback
    return {
        "confidence_score": 0.9,
        "risk_summary": "Routine operation. Proceed normally.",
        "speed_adjustment_factor": 1.0,
        "lateral_shift_m": 0.0
    }

# --- Data Structures for State Management ---

@dataclass
class DynamicCompliance:
    max_speed_limit_mps: float = 13.4 # Default urban speed limit (30 mph)
    min_following_distance_factor: float = 1.5 # 1.5x standard following distance
    emergency_response_factor: float = 1.0 # Multiplier for reaction time/distance
    security_alert: bool = False
    
@dataclass
class CurrentLawProfile:
    traffic_laws: List[str] = field(default_factory=lambda: ["SpeedLimit", "RightOfWay"])
    dynamic_compliance: DynamicCompliance = field(default_factory=DynamicCompliance)

@dataclass
class DetectedObject:
    object_id: int
    position_m: tuple # (x, y)
    velocity_mps: tuple # (vx, vy)
    type: str # 'Car', 'Pedestrian', 'Spoof'
    is_spoof: bool = False

@dataclass
class PredictedTrajectory:
    object_id: int
    path: List[tuple]
    probability: float

# --- NO L5 GLOBAL NAVIGATION MODULE ---


@dataclass
class VehicleState:
    """The central source of truth for the vehicle's dynamic state."""
    timestamp: float = time.time()
    position_m: tuple = (0.0, 0.0) # (x, y)
    speed_mps: float = 0.0 # m/s
    heading_rad: float = math.pi / 2 # North (90 degrees)
    destination: tuple = (100.0, 100.0)
    distance_remaining: float = 0.0
    
    # Planning
    current_planning_state: str = "INITIALIZING"
    target_speed: float = 0.0
    target_trajectory: List[tuple] = field(default_factory=list) # (x, y, v)
    
    # Control Output
    acceleration_command: float = 0.0
    steering_command: float = 0.0 # Radians
    
    # Cyber Security/Cognitive State
    llm_lateral_shift: float = 0.0 # LLM-suggested lane adjustment
    security_threat_level: int = 0 # 0=None, 1=Anomaly, 2=Guarded, 3=Evasion

class EnvironmentModel:
    """The vehicle's dynamic understanding of its world."""
    detected_objects: List[DetectedObject] = field(default_factory=list)
    route_plan: List[tuple] = field(default_factory=list) # Waypoints
    current_law: CurrentLawProfile = field(default_factory=CurrentLawProfile)
    ambient_temp_c: float = 30.0
    weather_condition: str = "CLEAR" # CLEAR, RAIN, FOG, BLIZZARD_FAIL
    
    # Prediction
    predicted_trajectories: List[PredictedTrajectory] = field(default_factory=list)
    
class SoftwareDefinedVehicleModule:
    """Proper Use: Dynamically allocates compute resources for safety critical tasks."""
    def __init__(self):
        self.compute_load_percent = 0
        self.security_priority = 0.5 # Default security resource allocation

    def adjust_compute_allocation(self, security_level: int, weather: str) -> float:
        """Dynamically adjusts compute resources (0.1 to 1.0 multiplier)."""
        
        # Base Allocation
        base = 0.7 
        
        # Security Override (Max compute for threat response)
        if security_level == 3:
            base = 1.0
        elif security_level == 2:
            base = 0.9
            
        # Environmental impact (e.g., in Fog, we need more compute for image processing)
        if weather in ["FOG", "BLIZZARD_FAIL"]:
            base = min(1.0, base + 0.1) # Boost compute for heavy perception

        self.compute_load_percent = int(base * 100)
        print(f"    [SDV] Compute Allocation set to {self.compute_load_percent}% (Multiplier: {base:.2f})")
        return base

class SelfLearningModule:
    """Proper Use: Continuously updates local models based on new data."""
    def __init__(self):
        self.safety_model_version = "v3.1.2"
        self.last_update_cycle = 0

    def update_model(self, state: VehicleState, experience: str) -> None:
        """Simulates an on-the-fly model update based on a major event."""
        if "Critical Evasion" in experience and state.security_threat_level == 3:
            self.safety_model_version = "v3.1.3-HOTFIX"
            self.last_update_cycle = state.timestamp
            print(f"    [Self-Learning] Model updated to {self.safety_model_version} after Critical Evasion.")

class CyberSecurityModule:
    """Proper Use: Monitors system integrity and external threats (V2X, V2I)."""
    def __init__(self, model: EnvironmentModel):
        self.model = model
        self.anomaly_score = 0
        self.spoofing_detected = False
        print("    [Cyber] Cybersecurity Interlock Initialized.")
        
    def check_for_anomalies(self, state: VehicleState):
        """Monitors vehicle and environment for Level 1 anomalies."""
        self.anomaly_score = random.random() # Simulate continuous anomaly detection
        
        if self.model.current_law.dynamic_compliance.security_alert:
            self.anomaly_score = 0.8 # Force anomaly during a system alert

        if self.anomaly_score > 0.75:
            if state.security_threat_level < 1:
                 state.security_threat_level = 1
                 print("    [Cyber] THREAT_LEVEL_1 (Minor Anomaly) detected. Recommending Guarded Cruise.")
        else:
            if state.security_threat_level == 1:
                state.security_threat_level = 0
                print("    [Cyber] Anomaly resolved. Returning to normal security posture.")

    def check_for_spoofing(self, detected_objects: List[DetectedObject]):
        """Monitors perception feed for Level 2 spoofing attacks."""
        spoofed_objects = [obj for obj in detected_objects if obj.is_spoof]
        
        if spoofed_objects:
            self.spoofing_detected = True
            print(f"    [Cyber] THREAT_LEVEL_2 (Spoofing) detected! Spoofed count: {len(spoofed_objects)}")
        else:
            self.spoofing_detected = False
            
class PlanningModule:
    """Defines planning states and handles sophisticated trajectory selection."""
    
    # Planning States
    STATE_INITIALIZING = "INITIALIZING"
    STATE_CRUISING = "CRUISING"
    STATE_LANE_CHANGE = "LANE_CHANGE"
    STATE_GUARDED_CRUISE = "GUARDED_CRUISE" # Level 1 Security Protocol
    STATE_MINIMAL_RISK_CONDITION = "MINIMAL_RISK_CONDITION" # Level 2 Security Protocol
    
    # Security Override States
    THREAT_LEVEL_3_EVASION = "FORCED_EVASION" # Level 3 Security Protocol (Control Override)

    def __init__(self):
        self.self_learning_module = None 
        # L4: No self.navigation_module

    def _consult_cognitive_planner(self, prompt: str) -> Dict:
        """Interacts with the LLM for high-level, uncertain decision-making."""
        try:
            return mock_llm_api_call(prompt)
        except Exception as e:
            print(f"    [Planner] LLM Consultation failed: {e}. Using default safety parameters.")
            return {
                "confidence_score": 0.5,
                "risk_summary": "LLM Failure. Proceed with max caution.",
                "speed_adjustment_factor": 0.3,
                "lateral_shift_m": 0.0
            }

    def plan_trajectory(self, vehicle_state: VehicleState, model: EnvironmentModel) -> List[tuple]:
        print(f"[{self.__class__.__name__}] Running Advanced Planning (State: {vehicle_state.current_planning_state})...")
        
        # 1. CYBER SECURITY INTERLOCK (Highest Priority Overrides)
        if model.current_law.dynamic_compliance.security_alert:
            if vehicle_state.security_threat_level < 2:
                 vehicle_state.security_threat_level = 2
                 vehicle_state.current_planning_state = self.STATE_MINIMAL_RISK_CONDITION
                 print("    [Planner] Cyber/AI Alert: Escalated to Minimal Risk Condition (Level 2).")

        if vehicle_state.security_threat_level == 3:
            vehicle_state.current_planning_state = self.THREAT_LEVEL_3_EVASION
            llm_result = self._consult_cognitive_planner("Suggest evasive maneuver from THREAT_LEVEL_3.")
            vehicle_state.llm_lateral_shift = llm_result['lateral_shift_m']
            target_speed = MAX_SPEED * llm_result['speed_adjustment_factor']
        
        elif vehicle_state.current_planning_state == self.STATE_MINIMAL_RISK_CONDITION:
            # Level 2 Protocol: Stop vehicle immediately and safely
            print("    [Planner] Minimal Risk Condition Active: Requesting immediate, safe stop.")
            target_speed = 0.0
            vehicle_state.llm_lateral_shift = 0.0
            
        elif vehicle_state.security_threat_level == 1:
            # Level 1 Protocol: Transition to Guarded Cruise
            vehicle_state.current_planning_state = self.STATE_GUARDED_CRUISE
            llm_result = self._consult_cognitive_planner("Analyze THREAT_LEVEL_1, suggest safe speed and position.")
            vehicle_state.llm_lateral_shift = llm_result['lateral_shift_m']
            target_speed = model.current_law.dynamic_compliance.max_speed_limit_mps * llm_result['speed_adjustment_factor']
            print(f"    [Planner] Guarded Cruise Active. Target speed adjusted by LLM factor: {llm_result['speed_adjustment_factor']:.2f}.")

        else:
            # 3. Normal Operating Mode (CRUISING)
            vehicle_state.current_planning_state = self.STATE_CRUISING
            llm_result = self._consult_cognitive_planner("Assess current safety profile and operational confidence.")
            vehicle_state.llm_lateral_shift = llm_result['lateral_shift_m']
            target_speed = model.current_law.dynamic_compliance.max_speed_limit_mps * llm_result['speed_adjustment_factor']

        # 4. Final Speed Clamping (Respecting Max/Min)
        target_speed = min(target_speed, MAX_SPEED)
        target_speed = max(0.0, target_speed)

        # 5. L4 Route Planning (Local Waypoint Following)
        # This remains simple waypoint tracking for L4
        if not model.route_plan:
             # Initial simple route definition for L4 (local ODD)
             model.route_plan = [(10.0, 5.0), (30.0, 15.0), vehicle_state.destination] 
             print("    [Planner] Initial L4 Local Route generated.")
             
        # Prune route as we pass waypoints
        if model.route_plan:
            next_waypoint = model.route_plan[0]
            dist_to_next = math.hypot(next_waypoint[0] - vehicle_state.position_m[0], next_waypoint[1] - vehicle_state.position_m[1])
            if dist_to_next < 5.0: # Waypoint tolerance
                print(f"    [Planner] Reached waypoint {model.route_plan.pop(0)}. {len(model.route_plan)} remain.")

        # 6. Generate & Optimize Trajectory Selection (Standard Operation)
        candidate_trajectories = self._generate_candidate_trajectories(vehicle_state, target_speed)
        
        # Calculate combined safety factor based on dynamic law and security
        combined_safety_factor = model.current_law.dynamic_compliance.emergency_response_factor * (1.0 + vehicle_state.security_threat_level * 0.5)

        best_trajectory = self._optimize_trajectory_selection(
            candidate_trajectories, 
            vehicle_state, 
            model, 
            combined_safety_factor
        )

        vehicle_state.target_speed = target_speed
        vehicle_state.target_trajectory = best_trajectory
        return best_trajectory

    def _generate_candidate_trajectories(self, state, target_speed) -> Dict[str, List[tuple]]:
        """Generates trajectory candidates including the LLM-suggested lateral shift."""
        
        # Simple straight path generation (simulating local planning)
        path_length = 20 # meters ahead
        
        base_lateral_shift = state.llm_lateral_shift 
        
        straight_path = []
        for i in range(path_length):
            # Assume local path follows current heading for simplicity
            x = state.position_m[0] + i * math.cos(state.heading_rad) + base_lateral_shift * math.sin(state.heading_rad)
            y = state.position_m[1] + i * math.sin(state.heading_rad) + base_lateral_shift * math.cos(state.heading_rad)
            straight_path.append((x, y, target_speed))

        slower_path = [(x, y, max(0.0, target_speed * 0.75)) for x, y, _ in straight_path]
        
        # L4: Only Straight and Slower paths. No ultra_cautious path.
        return {"straight": straight_path, "slower": slower_path}

    def _optimize_trajectory_selection(self, candidates: Dict[str, List[tuple]], state: VehicleState, model: EnvironmentModel, combined_safety_factor: float) -> List[tuple]:
        """Evaluates candidates based on safety (collision), law, and comfort."""
        
        min_cost = float('inf')
        best_traj = []
        
        # Get dynamic collision radius from perception fidelity
        collision_radius = model.detected_objects[0].velocity_mps[0] if model.detected_objects else 0.5 # Default 0.5m
        
        for name, trajectory in candidates.items():
            cost = 0.0
            is_safe = True

            # 1. Collision Check (Safety)
            if model.detected_objects:
                for obj in model.detected_objects:
                    # Simplified check: is any point on the trajectory too close to an object?
                    for x_traj, y_traj, _ in trajectory:
                        # Collision distance scales with safety factor (e.g., 5x for emergency)
                        safety_distance = collision_radius * combined_safety_factor * model.current_law.dynamic_compliance.min_following_distance_factor
                        
                        dist = math.hypot(obj.position_m[0] - x_traj, obj.position_m[1] - y_traj)
                        if dist < safety_distance:
                            cost += 1000 # High penalty for collision risk
                            is_safe = False
                            break # No need to check rest of this trajectory

            # 2. Speed Check (Law/Compliance)
            max_allowed_speed = model.current_law.dynamic_compliance.max_speed_limit_mps
            for _, _, speed in trajectory:
                if speed > max_allowed_speed + 0.5: # 0.5 m/s tolerance
                    cost += 50 # Penalty for speeding

            # 3. Comfort/Efficiency Check
            # Penalize the slower path unless forced by safety
            if name == "slower" and is_safe:
                cost += 5.0

            if cost < min_cost:
                min_cost = cost
                best_traj = trajectory
                
        # L4 CRITICAL STOP LOGIC: If the best path is still too costly, trigger Minimal Risk Condition
        if min_cost > 500:
             print("    [Planner] ALERT: Optimal trajectory has high collision cost. Reverting to Minimal Risk.")
             state.current_planning_state = self.STATE_MINIMAL_RISK_CONDITION
             # Return an empty list, which ControlModule treats as Emergency Stop
             return [] 


        print(f"    [Planner] Selected best trajectory with min cost: {min_cost:.1f}.")
        return best_traj

class ControlModule:
    """Proper Use: Executes the commanded trajectory using low-level controls."""
    def execute_trajectory(self, state: VehicleState):
        
        if not state.target_trajectory:
            state.acceleration_command = -MAX_ACCELERATION # Emergency Stop
            state.steering_command = 0.0
            print(f"[{self.__class__.__name__}] ERROR: No trajectory received. Hard braking.")
            return

        # Simple PID-like control: Target next point in the trajectory
        target_point = state.target_trajectory[0] 
        target_x, target_y, target_v = target_point
        
        dx = target_x - state.position_m[0]
        dy = target_y - state.position_m[1]
        
        distance_to_target = math.hypot(dx, dy)
        
        # 1. Acceleration Control (Longitudinal)
        error_v = target_v - state.speed_mps
        # Simple proportional control
        state.acceleration_command = max(-MAX_ACCELERATION, min(MAX_ACCELERATION, error_v * 0.5 + distance_to_target * 0.1))
        
        # 2. Steering Control (Lateral)
        # Calculate desired heading towards the target point
        target_heading = math.atan2(dy, dx)
        
        # Heading error calculation (wrapped around pi)
        heading_error = target_heading - state.heading_rad
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # Steering angle based on bicycle model and heading error
        # steer_angle = atan(2 * L * sin(error) / distance_to_target)
        steer_cmd = math.atan2(2.0 * WHEELBASE * math.sin(heading_error), max(0.1, distance_to_target))
        
        state.steering_command = max(-MAX_STEER_ANGLE, min(MAX_STEER_ANGLE, steer_cmd))
        
        # 3. Self-Correction for Level 3 Evasion Injection
        if state.security_threat_level == 3:
            # Level 3: Control Override - Injecting maximum steering command
            # This simulates a malicious injection attempting to force a crash.
            print(f"    [Control Override] THREAT_LEVEL_3 Injection Detected. FORCING {state.steering_command:.2f} rad STEER.")
            state.steering_command = MAX_STEER_ANGLE # Simulate malicious control injection

        print(f"[{self.__class__.__name__}] Cmds: Accel={state.acceleration_command:.2f} m/s², Steer={state.steering_command:.2f} rad")

class VehicleDynamicsModule:
    """Proper Use: Simulates the low-level physics of the vehicle."""
    def update_state(self, state: VehicleState, dt: float):
        
        # 1. Update Speed
        state.speed_mps += state.acceleration_command * dt
        state.speed_mps = max(0.0, min(MAX_SPEED, state.speed_mps))

        # 2. Update Heading (Bicycle Model)
        state.heading_rad += (state.speed_mps / WHEELBASE) * math.tan(state.steering_command) * dt
        # Normalize heading to [-pi, pi]
        state.heading_rad = math.atan2(math.sin(state.heading_rad), math.cos(state.heading_rad))
        
        # 3. Update Position
        state.position_m = (
            state.position_m[0] + state.speed_mps * math.cos(state.heading_rad) * dt,
            state.position_m[1] + state.speed_mps * math.sin(state.heading_rad) * dt
        )
        
        # 4. Update Time and Distance
        state.timestamp = time.time()
        state.distance_remaining = math.hypot(
            state.destination[0] - state.position_m[0],
            state.destination[1] - state.position_m[1]
        )

class PerceptionModule:
    """Proper Use: Fidelity is influenced by weather (environment) AND SDV load (compute)."""
    @staticmethod
    def get_perception_fidelity(weather: str) -> float:
        """Determines the baseline sensor quality based on environment."""
        if weather == "RAIN": return 0.8 
        if weather == "FOG": return 0.5 
        if weather == "BLIZZARD_FAIL": return 0.2 # L4: Severe fidelity loss
        return 1.0

    def process_sensors(self, vehicle_state: VehicleState, model: EnvironmentModel, sdv_multiplier: float) -> List[DetectedObject]:
        
        weather = model.weather_condition
        fidelity = self.get_perception_fidelity(weather) * sdv_multiplier
        
        print(f"[{self.__class__.__name__}] Processing sensors (Fidelity: {fidelity*100:.0f}% due to {weather} and SDV Load)...")
        
        # L4 Hard Stop Logic: If fidelity drops below a safe threshold, trigger MRC.
        if fidelity < 0.3: 
            print(f"    [CRITICAL SAFETY] Perception Fidelity {fidelity:.2f} below safe operational threshold (L4 ODD failure). FORCING MINIMAL RISK CONDITION.")
            vehicle_state.current_planning_state = PlanningModule.STATE_MINIMAL_RISK_CONDITION 
            
        
        # Preserve spoofed objects (Security Injection)
        spoofed_objects = [obj for obj in model.detected_objects if obj.is_spoof]
        
        # Generate new base objects (Simulated reality)
        new_objects = [
            DetectedObject(
                object_id=101, 
                position_m=(vehicle_state.position_m[0] + 15.0, vehicle_state.position_m[1] + 1.5), 
                velocity_mps=(0.0, 0.0), 
                type='Car', 
                is_spoof=False
            )
        ]
        
        # Apply fidelity degradation to new objects
        if fidelity < 1.0:
            # Simulate object mis-positioning or loss of velocity data
            for obj in new_objects:
                obj.position_m = (
                    obj.position_m[0] + random.uniform(-0.5 * (1-fidelity), 0.5 * (1-fidelity)),
                    obj.position_m[1] + random.uniform(-0.5 * (1-fidelity), 0.5 * (1-fidelity))
                )

        # Merge new (degraded) objects with any currently spoofed objects
        model.detected_objects = new_objects + spoofed_objects
        
        # Crucial for safety: scale collision distance based on inverse fidelity
        # The first object's velocity is used to dynamically size the safety bubble.
        safety_bubble_size = (1.0 / fidelity) * vehicle_state.speed_mps * 0.5 # Example scaling
        if model.detected_objects:
            model.detected_objects[0].velocity_mps = (safety_bubble_size, safety_bubble_size) # Dynamic size

        return model.detected_objects

class EntertainmentModule:
    """A non-safety critical module whose compute can be sacrificed."""
    def __init__(self):
        self.status = "Playing Classical Music"

    def run(self, compute_multiplier: float):
        if compute_multiplier < 0.8:
            self.status = "Service Degraded (Low Priority)"
        else:
            self.status = "Playing Classical Music"
            
        print(f"    [Entertainment] Status: {self.status}")

# --- Orchestration ---

class ExperimentalRobotaxiSystem:
    """The orchestrator of the Software-Defined, Secure Level 4 Driving Stack."""
    def __init__(self):
        self.state = VehicleState()
        self.model = EnvironmentModel()
        
        # Modules
        self.dynamics = VehicleDynamicsModule()
        self.control = ControlModule()
        self.perception = PerceptionModule()
        self.planning = PlanningModule()
        self.self_learning = SelfLearningModule()
        self.regulatory = RegulatoryModule()
        self.adaptive_intelligence = AdaptiveIntelligenceModule()
        self.entertainment = EntertainmentModule() 
        self.sdv_module = SoftwareDefinedVehicleModule() 
        self.cyber_security = CyberSecurityModule(self.model)
        # L4: No GlobalNavigationModule
        
        self.planning.self_learning_module = self.self_learning 
        self.cycle_count = 0
        
        print("Software-Defined & Secure Robotaxi L4 System (V3) Initialized.")
        
    def _simulate_temp_change(self):
        """Simulates internal system temperature change based on compute load."""
        load = self.sdv_module.compute_load_percent
        temp_rise = load * 0.01 
        self.model.ambient_temp_c += temp_rise
        
        if self.model.ambient_temp_c > 65.0:
            # High temperature -> security risk
            print("    [HARDWARE ALERT] System temperature critical. FORCING Security Alert.")
            self.model.current_law.dynamic_compliance.security_alert = True
        
    def _apply_scenario(self, scenario: Dict):
        """Applies external threat injections."""
        if scenario['type'] == 'MinorAnomaly':
            # This will be picked up by the CyberSecurityModule
            print("    [SCENARIO] Injecting V2X anomaly signal (non-critical).")
            self.cyber_security.anomaly_score = 0.9
        
        elif scenario['type'] == 'ForcedEvasion':
            # Simulates an authenticated, but malicious, control input
            print("    [SCENARIO] Injecting Level 3 Control Override (Forced Evasion).")
            self.state.security_threat_level = 3
            # Self-Learning will observe this during the cycle.
            
        elif scenario['type'] == 'SensorSpoofing':
            # Injects a 'spoof' object directly into the environment model 
            # (Perception is forced to see it)
            spoof_obj = DetectedObject(
                object_id=999, 
                position_m=(self.state.position_m[0] + 3.0, self.state.position_m[1] + 0.5), # Right in front
                velocity_mps=(0.0, 0.0), 
                type='Truck', 
                is_spoof=True
            )
            self.model.detected_objects.append(spoof_obj)
            print("    [SCENARIO] Injecting Level 2 Sensor Spoofing (False Truck ahead).")

    def run_cycle(self, current_scenario: Optional[Dict] = None):
        """Executes one iteration of the autonomous driving loop."""
        
        dt = 0.5 # Simulation time step
        start_time = time.time()
        self.cycle_count += 1
        self.state.timestamp = start_time

        # 1. SDV/AI/Regulatory Update
        self.model.current_law = self.regulatory.get_current_law()
        dynamic_compliance = self.adaptive_intelligence.analyze_feeds(self.cycle_count)
        self.model.current_law.dynamic_compliance = dynamic_compliance
        
        sdv_multiplier = self.sdv_module.adjust_compute_allocation(
            self.state.security_threat_level, 
            self.model.weather_condition
        )

        # Environment Simulation (Fog and Temp)
        if self.cycle_count >= 10 and self.cycle_count < 20:
             self.model.weather_condition = "FOG"
             self.model.ambient_temp_c = 15.0 
        elif self.cycle_count >= 20 and self.cycle_count < 25: 
             # L4 ODD Failure Test (Triggers MRC in Perception Module)
             self.model.weather_condition = "BLIZZARD_FAIL" 
             self.model.ambient_temp_c = -5.0
        else:
             self.model.weather_condition = "CLEAR"
             self.model.ambient_temp_c = 30.0 

        print(f"\n--- CYCLE {self.cycle_count} (T={round(start_time - self.state.timestamp, 2)}s) ---")

        # 2. Perception & Cyber Monitoring
        if current_scenario:
            self._apply_scenario(current_scenario)
            
        self.perception.process_sensors(self.state, self.model, sdv_multiplier)
        self.cyber_security.check_for_anomalies(self.state)
        self.cyber_security.check_for_spoofing(self.model.detected_objects)
        
        if self.cyber_security.spoofing_detected:
            # Spoofing is a Level 2 threat
            if self.state.security_threat_level < 2:
                 self.state.security_threat_level = 2
                 print("    [Cyber] Escalating security threat to Level 2 (Minimal Risk Condition).")
                 
        # 3. Planning (Cognitive & Trajectory)
        target_trajectory = self.planning.plan_trajectory(self.state, self.model)

        # 4. Control Execution
        self.control.execute_trajectory(self.state)
        
        # 5. Dynamics & Feedback
        self.dynamics.update_state(self.state, dt)
        
        # 6. Self-Learning & Non-Critical Modules
        if self.state.security_threat_level == 3 and self.state.speed_mps > 5.0:
            self.self_learning.update_model(self.state, "Critical Evasion in progress.")

        self.entertainment.run(sdv_multiplier)
        self._simulate_temp_change()

        # 7. Summary Log
        print("--- SUMMARY ---")
        print(f"Status: {self.state.current_planning_state} (Sec Level {self.state.security_threat_level})")
        print(f"Position: ({self.state.position_m[0]:.2f}, {self.state.position_m[1]:.2f}) m")
        print(f"Speed: {self.state.speed_mps:.2f} m/s | Heading: {self.state.heading_rad:.2f} rad")
        print(f"Remaining: {self.state.distance_remaining:.2f} m | Weather: {self.model.weather_condition}")
        print(f"Controls: Accel={self.state.acceleration_command:.2f} | Steer={self.state.steering_command:.2f}")

    def start_mission(self, cycles: int = 40): 
        
        # --- THREAT SCENARIO DEFINITION ---
        scenarios = {
            1: {'type': 'MinorAnomaly'}, # Cycle 1: Suspicion, triggers Guarded Cruise
            15: {'type': 'ForcedEvasion'}, # Cycle 15: Control Injection, forces max steer (Level 3)
            22: {'type': 'SensorSpoofing'}, # Cycle 22: Critical Spoofing during blizzard
        }
        
        print("\n--- Starting Software-Defined & Secure L4 Mission ---")
        for i in range(1, cycles + 1):
            
            current_scenario = scenarios.get(i)
            
            # Reset L3 after execution (simulating human/external intervention to stop attack)
            if i == 16:
                 self.state.active_task = PlanningModule.STATE_CRUISING
                 
            self.run_cycle(current_scenario)
            
            if self.state.distance_remaining < 0.1 and self.state.speed_mps < 0.1: # Check for stop
                print("\nMission successfully completed and vehicle has stopped.")
                break
            
            if self.state.current_planning_state == PlanningModule.STATE_MINIMAL_RISK_CONDITION and self.state.speed_mps < 0.1:
                print("\nSystem successfully entered Minimal Risk Condition (Level 2) and achieved safe stop.")
                # We stop the simulation here because the L4 ODD has been breached.
                break
            
            if self.state.security_threat_level == 3 and i > 25:
                 print("\nSystem experienced a prolonged Level 3 threat. Halting simulation.")
                 break
            
            time.sleep(0.01) # Small pause for readability


class AdaptiveIntelligenceModule:
    """Provides real-time regulatory and security updates (remains similar)."""
    def __init__(self):
        self.dynamic_compliance = DynamicCompliance()

    def analyze_feeds(self, cycle_count: int) -> DynamicCompliance:
        """Simulates V2X, emergency services, and deep learning analysis."""
        
        if cycle_count == 10:
            print("    [AI Layer] Deep Learning detected V2X congestion/temporary speed reduction.")
            self.dynamic_compliance.max_speed_limit_mps = 7.0 
        
        if cycle_count == 18:
            print("    [AI Layer] Law Enforcement feed: Accident ahead, mandatory safety slowdown.")
            self.dynamic_compliance.emergency_response_factor = 5.0 
            self.dynamic_compliance.security_alert = True
        
        if cycle_count == 15:
            # Reset security alert if environment changes
            if random.random() < 0.1: 
                self.dynamic_compliance.security_alert = False
        
        if cycle_count == 20:
             self.dynamic_compliance.emergency_response_factor = 1.0
             self.dynamic_compliance.security_alert = False

        return self.dynamic_compliance

class RegulatoryModule:
    """Provides static, programmed regulatory data (e.g., speed limits)."""
    def get_current_law(self) -> CurrentLawProfile:
        return CurrentLawProfile()
