import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import random
from abc import ABCMeta, abstractmethod
import json

# --- Production Libraries (must be installed on the Android compute unit) ---
# We will show these as comments for compatibility, but they are required.
# import requests # For making HTTP requests to Gemini and Google Maps
# import can      # For interfacing with the vehicle's CAN bus
# import android.hardware.camera2 # For accessing the NDK camera
# import android.location        # For accessing the Android Location/GPS API

# --- 1. Vehicle Agnostic Profile ---
@dataclass
class VehicleProfile:
    """A data-driven profile defining the physical characteristics of a vehicle."""
    name: str = "Production Vehicle (Camera/Radar Only)"
    wheelbase: float = 2.8 # meters
    max_steer_angle_rad: float = 0.61 # Radians
    max_acceleration_mps2: float = 5.0 # m/s^2
    max_deceleration_mps2: float = -8.0 # m/s^2
    max_speed_mps: float = 40.0 # m/s
    
    # --- HARDWARE CONSTRAINT ---
    has_lidar: bool = False # No additional hardware
    
    # --- CAN BUS Configuration ---
    can_interface: str = 'slcan0'
    can_steer_id: int = 0x0F1     # Arbitration ID for steering commands
    can_throttle_id: int = 0x0F2  # Arbitration ID for throttle/brake
    can_feedback_id: int = 0x1A3   # Arbitration ID for wheel speed feedback

# --- 2. Abstract Base Classes (The "Agile" Interfaces) ---
class AbstractPerception(metaclass=ABCMeta):
    @abstractmethod
    def process_sensors(self) -> List['DetectedObject']:
        """Get processed sensor data from the Android OS."""
        pass
        
    @abstractmethod
    def get_perception_fidelity(self) -> float:
        """Get the current fidelity based on weather/lighting."""
        pass

class AbstractNavigator(metaclass=ABCMeta):
    @abstractmethod
    def fetch_route_and_laws(self, origin: str, destination: str) -> bool:
        """Fetches the complete route. Returns True on success."""
        pass

    @abstractmethod
    def get_current_law(self, state: 'VehicleState') -> 'CurrentLawProfile':
        """Gets the law for the *current* segment of the route."""
        pass
        
    @abstractmethod
    def get_route_prompt_segment(self, state: 'VehicleState') -> str:
        """Gets the next 5 steps of the route as a string for the AI."""
        pass

class AbstractPlanner(metaclass=ABCMeta):
    @abstractmethod
    def get_cognitive_plan(self, state: 'VehicleState', model: 'EnvironmentModel', nav_prompt: str):
        """Contacts the Cloud AI to get a definitive driving plan."""
        pass

class AbstractController(metaclass=ABCMeta):
    @abstractmethod
    def execute_commands(self, state: 'VehicleState', profile: VehicleProfile):
        """Sends the AI's commands to the vehicle CAN bus."""
        pass
        
    @abstractmethod
    def send_safe_park_command(self, profile: VehicleProfile):
        """Sends a command to engage parking brake and neutral."""
        pass

class AbstractDynamicsReader(metaclass=ABCMeta):
    @abstractmethod
    def update_state_from_feedback(self, state: 'VehicleState', profile: VehicleProfile):
        """Reads CAN bus feedback to update the vehicle's state."""
        pass

class AbstractSDV(metaclass=ABCMeta):
    @abstractmethod
    def adjust_compute_allocation(self, vehicle_state: 'VehicleState', environment_model: 'EnvironmentModel') -> float:
        """Manages SoC and Network priority."""
        pass

class AbstractCyberSecurity(metaclass=ABCMeta):
    @abstractmethod
    def check_for_anomalies(self, state: 'VehicleState'):
        pass
    
    @abstractmethod
    def validate_command(self, state: 'VehicleState', profile: VehicleProfile) -> bool:
        """Validates the AI's command against low-level safety rules."""
        pass

# --- 3. Data Structures ---

@dataclass
class CurrentLawProfile:
    max_speed_limit_mps: float = 13.4 
    current_road_name: str = "Unknown"
    upcoming_maneuver: str = "Continue straight"

@dataclass
class DetectedObject:
    object_id: int
    type: str 
    position_m: tuple 
    velocity_mps: tuple 
    confidence: float # Confidence from 0.0 to 1.0

@dataclass
class VehicleState:
    timestamp: float = field(default_factory=time.time)
    position_m: tuple = (0.0, 0.0) # Local ECEF (for relative calcs)
    gps: tuple = (34.0522, -118.2437) # (lat, lon) from Android GPS
    speed_mps: float = 0.0 # From CAN bus feedback
    heading_rad: float = math.pi / 2 # From CAN/IMU feedback
    destination: str = "" # User-provided destination
    
    current_planning_state: str = "STANDBY"
    
    # AI-Requested Commands (to be sent to CAN)
    target_acceleration_request: float = 0.0
    target_steering_request: float = 0.0
    
    # Final Applied Commands (after safety validation)
    acceleration_command: float = 0.0
    steering_command: float = 0.0
    
    security_threat_level: int = 0 

class EnvironmentModel:
    detected_objects: List[DetectedObject] = field(default_factory=list)
    current_law: CurrentLawProfile = field(default_factory=CurrentLawProfile)
    weather_condition: str = "CLEAR" # From external weather API
    
# --- 4. Concrete Production Module Implementations ---

class GoogleMapsNavigator(AbstractNavigator):
    """
    Production implementation that uses the Google Maps Directions API
    for navigation and regulatory information.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://maps.googleapis.com/maps/api/directions/json"
        self.route_steps: List[Dict] = []
        self.current_step_index = 0
        print("[Navigator] Google Maps API Initialized.")

    def fetch_route_and_laws(self, origin: str, destination: str) -> bool:
        """Makes a real API call to Google Maps to get the route."""
        # This function would use 'requests' in a real Python env
        # params = {
        #     'origin': origin,
        #     'destination': destination,
        #     'key': self.api_key,
        #     'mode': 'driving'
        # }
        # response = requests.get(self.api_url, params=params)
        # if not response.ok or route_data.get('status') != 'OK':
        #     print(f"[Navigator] ERROR: Failed to fetch route: {route_data.get('status')}")
        #     return False
        # route_data = response.json()
        
        # --- MOCKING THE API RESPONSE for this environment ---
        print(f"[Navigator] Fetching route from {origin} to {destination}...")
        self.route_steps = [
            {'html_instructions': 'Head north on Main St', 'distance': 500, 'speed_limit_mps': 13.4},
            {'html_instructions': 'Turn left onto 1st Ave', 'distance': 1200, 'speed_limit_mps': 11.2},
            {'html_instructions': 'Merge onto I-5 N', 'distance': 10000, 'speed_limit_mps': 29.0},
            {'html_instructions': 'Arrive at Disneyland', 'distance': 100, 'speed_limit_mps': 5.0},
        ]
        self.current_step_index = 0
        print(f"[Navigator] Route acquired with {len(self.route_steps)} steps.")
        return True # <-- Return True on success

    def _get_current_step(self, state: 'VehicleState') -> Dict:
        # In a real system, this would use GPS to find the closest step
        if self.current_step_index >= len(self.route_steps):
            return {'html_instructions': 'Arrived at destination', 'distance': 0, 'speed_limit_mps': 0.0}
        
        # Mock advancing the route
        if state.speed_mps > 1 and random.random() > 0.98:
            self.current_step_index = min(self.current_step_index + 1, len(self.route_steps) - 1)
            
        return self.route_steps[self.current_step_index]

    def get_current_law(self, state: VehicleState) -> CurrentLawProfile:
        """Gets law and maneuver for the current route segment."""
        step = self._get_current_step(state)
        return CurrentLawProfile(
            max_speed_limit_mps=step.get('speed_limit_mps', 13.4),
            current_road_name="Mock Road Name",
            upcoming_maneuver=step.get('html_instructions', 'Continue')
        )
        
    def get_route_prompt_segment(self, state: VehicleState) -> str:
        if not self.route_steps:
            return "No route is active."
        start = self.current_step_index
        end = min(start + 3, len(self.route_steps))
        steps_text = "; ".join([step['html_instructions'] for step in self.route_steps[start:end]])
        return f"Current Maneuver: {self.route_steps[start]['html_instructions']}. Next Steps: {steps_text}"


class GeminiCognitivePlanner(AbstractPlanner):
    """
    Production implementation that calls the Gemini API for cognitive planning.
    It does NOT plan locally. It formats the prompt and parses the response.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={self.api_key}"
        self.self_learning_module: Optional[AbstractSelfLearning] = None
        
        # This JSON schema is CRITICAL for a production system.
        self.response_schema = {
            "type": "OBJECT",
            "properties": {
                "risk_summary": {"type": "STRING"},
                "target_acceleration_mps2": {"type": "NUMBER"},
                "target_steering_angle_rad": {"type": "NUMBER"},
                "confidence": {"type": "NUMBER"}
            },
            "required": ["risk_summary", "target_acceleration_mps2", "target_steering_angle_rad", "confidence"]
        }

    def _format_prompt(self, state: VehicleState, model: EnvironmentModel, nav_prompt: str) -> str:
        """Creates the rich text prompt for the Gemini API."""
        
        # Compress sensor data into text
        sensor_text = "No objects detected."
        if model.detected_objects:
            sensor_text = "Detected Objects: "
            for obj in model.detected_objects:
                sensor_text += f"[ID: {obj.object_id}, Type: {obj.type}, Pos: ({obj.position_m[0]:.1f}, {obj.position_m[1]:.1f}), Conf: {obj.confidence:.2f}]; "

        prompt = f"""
        Act as the cognitive core for an L5 autonomous vehicle.
        My current state:
        - Speed: {state.speed_mps:.2f} m/s
        - GPS: {state.gps}
        - Security Level: {state.security_threat_level}
        - Weather: {model.weather_condition}
        
        My navigation plan:
        - {nav_prompt}
        - Legal Speed Limit: {model.current_law.max_speed_limit_mps:.2f} m/s

        My sensor data (Camera/Radar):
        - {sensor_text}

        Your task is to return a JSON object with the required schema.
        Analyze all inputs and provide a safe, legal, and smooth driving decision.
        If a 'LowConfidenceBlob' is detected, identify it and proceed.
        If a 'Pedestrian' is 'UNKNOWN', predict their intent.
        """
        return prompt

    def get_cognitive_plan(self, state: VehicleState, model: EnvironmentModel, nav_prompt: str):
        """Makes a real HTTP request to the Gemini API."""
        
        prompt = self._format_prompt(state, model, nav_prompt)
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": self.response_schema
            }
        }
        
        # --- Real API Call (Mocked for this env) ---
        # try:
        #     response = requests.post(self.api_url, json=payload)
        #     response.raise_for_status() # Raise exception for bad status
        #     json_response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        #     ai_plan = json.loads(json_response_text)
        # except Exception as e:
        #     print(f"[Planner] CRITICAL AI FAILURE: {e}. Activating Fallback Safety.")
        #     ai_plan = {"risk_summary": "AI API FAILED", "target_acceleration_mps2": -1.0, "target_steering_angle_rad": 0.0, "confidence": 0.0}
        
        # --- Mocking the API response for this environment ---
        ai_plan = {
            "risk_summary": "Nominal cruise. Low confidence blob resolved as plastic bag.",
            "target_acceleration_mps2": 1.5,
            "target_steering_angle_rad": 0.0,
            "confidence": 0.95
        }
        # --- End Mock ---

        print(f"    [Planner] AI Plan: {ai_plan['risk_summary']} (Accel: {ai_plan['target_acceleration_mps2']:.2f}, Steer: {ai_plan['target_steering_angle_rad']:.2f})")
        
        # Set the state requests
        state.target_acceleration_request = ai_plan['target_acceleration_mps2']
        state.target_steering_request = ai_plan['target_steering_angle_rad']
        
        if self.self_learning_module and ai_plan['confidence'] < 0.8:
            self.self_learning_module.generalize_novel_event(state, ai_plan['risk_summary'])


class AndroidCANInterface(AbstractController, AbstractDynamicsReader):
    """
    Production implementation that interfaces with the vehicle CAN bus.
    This replaces BOTH the Controller and Dynamics simulator.
    """
    def __init__(self, profile: VehicleProfile):
        self.bus = None
        self.profile = profile
        try:
            # self.bus = can.interface.Bus(channel=profile.can_interface, bustype='socketcan')
            print(f"[CAN] Hardware interface initialized on {profile.can_interface}.")
        except Exception as e:
            print(f"[CAN] WARN: Failed to initialize CAN bus ({e}). Running in virtual mode.")

    def _float_to_can_bytes(self, value, min_val, max_val):
        """Utility to convert a float (e.g., steer angle) to CAN bytes."""
        value_norm = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
        int_val = int(value_norm * 65535) # 16-bit
        return [(int_val >> 8) & 0xFF, int_val & 0xFF]

    def execute_commands(self, state: VehicleState, profile: VehicleProfile):
        """Sends the AI's commands to the vehicle CAN bus."""
        
        # 1. Send Steering Command
        steer_bytes = self._float_to_can_bytes(
            state.steering_command, 
            -profile.max_steer_angle_rad, 
            profile.max_steer_angle_rad
        )
        # steer_msg = can.Message(arbitration_id=profile.can_steer_id, data=steer_bytes, is_extended_id=False)
        
        # 2. Send Throttle/Brake Command
        accel_bytes = self._float_to_can_bytes(
            state.acceleration_command,
            profile.max_deceleration_mps2,
            profile.max_acceleration_mps2
        )
        # accel_msg = can.Message(arbitration_id=profile.can_throttle_id, data=accel_bytes, is_extended_id=False)
        
        try:
            # self.bus.send(steer_msg)
            # self.bus.send(accel_msg)
            pass # No-op in virtual mode
        except Exception as e:
            print(f"[CAN] ERROR sending command: {e}")
            
    def send_safe_park_command(self, profile: VehicleProfile):
        """NEW: Sends a command to engage parking brake and neutral."""
        print("[CAN] CRITICAL ERROR: Sending SAFE PARK command to CAN bus.")
        # 1. Send 0 acceleration
        accel_bytes = self._float_to_can_bytes(0.0, profile.max_deceleration_mps2, profile.max_acceleration_mps2)
        # 2. Send 0 steering
        steer_bytes = self._float_to_can_bytes(0.0, -profile.max_steer_angle_rad, profile.max_steer_angle_rad)
        # 3. Send Park command (e.g., data=[0x01])
        # ... (implementation specific)
        try:
            # self.bus.send(accel_msg)
            # self.bus.send(steer_msg)
            # self.bus.send(park_msg)
            pass
        except Exception as e:
            print(f"[CAN] ERROR sending SAFE PARK command: {e}")

    def update_state_from_feedback(self, state: VehicleState, profile: VehicleProfile):
        """Reads CAN bus feedback to update the vehicle's state."""
        try:
            # msg = self.bus.recv(timeout=0.01) # Non-blocking read
            # if msg and msg.arbitration_id == profile.can_feedback_id:
            #     # Mock parsing: 2 bytes for speed (m/s * 100)
            #     speed_raw = (msg.data[0] << 8) | msg.data[1]
            #     state.speed_mps = speed_raw / 100.0
            
            # --- Mocking Feedback (since we have no real bus) ---
            # Simulate the state slowly matching the command
            dt = 0.4 # Assuming this runs in the main loop
            
            # Only simulate speed if commands are non-zero (i.e., not in standby)
            if state.acceleration_command != 0 or state.steering_command != 0 or state.speed_mps > 0:
                state.speed_mps += (state.acceleration_command - state.speed_mps * 0.1) * dt
                state.speed_mps = max(0.0, state.speed_mps) # Ensure it doesn't go negative
            # --- End Mock ---
            pass
        
        except Exception as e:
            print(f"[CAN] ERROR reading feedback: {e}")
        
        # Also update GPS from the Android Sensor Suite (via proxy)
        state.gps, state.heading_rad = AndroidSensorSuite.get_android_location_feedback()


class AndroidSensorSuite(AbstractPerception):
    """
    Production implementation that interfaces with the Android OS
    to get Camera, Radar, and GPS data.
    """
    def __init__(self, profile: VehicleProfile):
        self.profile = profile
        self.weather = "CLEAR" # This would come from a weather API
        print("[Android] Sensor Suite Initialized (Camera, Radar, GPS).")
    
    @staticmethod
    def get_android_location_feedback() -> tuple:
        """Static method to get location data from the OS."""
        # try:
        #     # location = android.location.getLastKnownLocation()
        #     # return (location.getLatitude(), location.getLongitude()), location.getBearing()
        # except Exception as e:
        #     print(f"[Android] Failed to get real GPS: {e}")
        
        # Mocking GPS feedback
        return (34.0522 + random.random() * 0.001, -118.2437 + random.random() * 0.001), math.pi / 2

    def get_perception_fidelity(self) -> float:
        """Fidelity is HIGHLY dependent on weather without LiDAR."""
        if self.profile.has_lidar:
             if self.weather == "BLIZZARD_CRITICAL": return 0.9
             return 1.0
        
        if self.weather == "BLIZZARD_CRITICAL": return 0.1 # Camera useless
        if self.weather == "FOG": return 0.3 # Camera/Radar highly degraded
        if self.weather == "RAIN": return 0.6
        return 0.9 # Even "CLEAR" has lower confidence

    def process_sensors(self) -> List[DetectedObject]:
        """Gets data from Camera/Radar and does minimal local processing."""
        
        # 1. Get data from NDK/SDK
        # camera_frame = android.hardware.camera2.getFrame()
        # radar_points = android.hardware.radar.getPoints()
        
        # 2. Run a very lightweight, local pre-processor (e.g., YOLO-lite)
        # This processor is *designed* to be low-confidence.
        
        # 3. Mocked result of that local processor:
        fidelity = self.get_perception_fidelity()
        new_objects = [
            DetectedObject(101, (15.0, 1.5), (0.0, 0.0), 'Car', False, "PASSIVE", 0.9 * fidelity),
            DetectedObject(102, (5.0, -0.5), (0.0, 0.0), 'Pedestrian', False, "UNKNOWN", 0.8 * fidelity),
            DetectedObject(103, (20.0, 0.0), (0.0, 0.0), 'LowConfidenceBlob', False, "UNKNOWN", 0.4 * fidelity) # Triggers AI Escalation
        ]
        
        return new_objects

class AndroidComputeModule(AbstractSDV):
    """Manages compute AND network resources on a single SoC."""
    def __init__(self):
        self.compute_load_percent = 0
        self.network_priority = "NORMAL"

    def adjust_compute_allocation(self, vehicle_state: VehicleState, environment_model: EnvironmentModel) -> float:
        base = 0.8 
        if vehicle_state.security_threat_level >= 2:
            base = 1.0
            self.network_priority = "CRITICAL" # Prioritize 5G link for AI
            print("    [Android SoC] Network priority set to CRITICAL for AI planning.")
        else:
            self.network_priority = "NORMAL"
        
        weather_tax = 0.1 if environment_model.weather_condition in ["FOG", "BLIZZARD_CRITICAL"] else 0.0
        multiplier = min(1.0, base + weather_tax)
        self.compute_load_percent = int(multiplier * 100)
        return multiplier

class CyberSecurityModule(AbstractCyberSecurity):
    def __init__(self, model: EnvironmentModel):
        self.model = model
        self._anomaly_score = 0
        self._spoofing_detected = False
        
    def check_for_anomalies(self, state: VehicleState):
        """Monitors for CAN bus anomalies or API inconsistencies."""
        self._anomaly_score = random.random()
        if self._anomaly_score > 0.9:
            if state.security_threat_level < 1:
                 state.security_threat_level = 1
                 print("    [Cyber] THREAT_LEVEL_1 (CAN Anomaly) detected.")
        else:
            if state.security_threat_level == 1: state.security_threat_level = 0

    def validate_command(self, state: VehicleState, profile: VehicleProfile) -> bool:
        """Final safety check before commands are sent to CAN."""
        # Example rule: Don't allow max steer at max speed
        if state.speed_mps > (profile.max_speed_mps * 0.8) and \
           abs(state.target_steering_request) > (profile.max_steer_angle_rad * 0.5):
            print("[Cyber] SAFETY OVERRIDE: AI requested high steer at high speed. Clamping command.")
            state.target_steering_request = profile.max_steer_angle_rad * 0.5
            return False
            
        # TODO: Add Level 3 injection check
        
        # All checks passed
        return True

    def check_for_spoofing(self, detected_objects: List[DetectedObject]): pass # Handled by AI
    @property
    def spoofing_detected(self) -> bool: return self._spoofing_detected

class SelfLearningModule(AbstractSelfLearning):
    def __init__(self):
        self.safety_model_version = "L5-v1.0-Cloud"
    def generalize_novel_event(self, state: VehicleState, event_summary: str):
        # In this model, generalization happens in the cloud,
        # but we can log that it was triggered.
        print(f"    [Self-Learning] Flagged event '{event_summary}' for cloud model generalization.")

# --- 6. The Orchestrator (Agile) ---

class GlobalAutonomySystem:
    """The main orchestrator, running as the primary Python service on the Android SoC."""
    def __init__(self,
                 vehicle_profile: VehicleProfile,
                 perception: AbstractPerception,
                 planner: AbstractPlanner,
                 control: AbstractController,
                 dynamics_reader: AbstractDynamicsReader,
                 sdv_module: AbstractSDV,
                 cyber_security: AbstractCyberSecurity,
                 self_learning: AbstractSelfLearning,
                 navigator: AbstractNavigator
                 ):
        
        self.profile = vehicle_profile
        self.state = VehicleState()
        self.model = EnvironmentModel()
        
        # Injected Modules (following the "Agile" contracts)
        self.dynamics_reader = dynamics_reader
        self.control = control
        self.perception = perception
        self.planner = planner
        self.self_learning = self_learning
        self.navigator = navigator
        self.sdv_module = sdv_module
        self.cyber_security = cyber_security
        
        self.cycle_count = 0
        self.mission_active = False # NEW: Flag to control the main loop
        
        print(f"--- Global Autonomy L5 System Initialized for: {self.profile.name} ---")
        
    def set_mission(self, destination_query: str):
        """
        NEW: This is the entry point for the Android UI to start a trip.
        It's "fully automated" because it handles fetching the route.
        """
        print(f"[Core] Mission Received: Plotting route to '{destination_query}'...")
        self.state.destination = destination_query
        
        # 1. Get current location from hardware
        self.dynamics_reader.update_state_from_feedback(self.state, self.profile)
        origin_str = f"{self.state.gps[0]},{self.state.gps[1]}"
        
        # 2. Fetch route from Google Maps
        if self.navigator.fetch_route_and_laws(origin_str, destination_query):
            self.mission_active = True
            self.state.current_planning_state = "GLOBAL_GENERALIZATION"
            print("[Core] Route locked. Mission is active. Engaging drive systems.")
        else:
            print(f"[Core] CRITICAL: Failed to plot route to {destination_query}. Mission aborted.")
            self.mission_active = False

    def _apply_scenario(self, scenario: Dict):
        if scenario['type'] == 'CriticalWeather':
            self.model.weather_condition = "FOG" # Camera-only will struggle
            print("    [SCENARIO] Injecting FOG (Camera/Radar stress test).")

    def run_cycle(self, current_scenario: Optional[Dict] = None):
        dt = 0.4
        self.cycle_count += 1
        
        print(f"\n--- CYCLE {self.cycle_count} (T={time.time():.0f}) ---")
        
        if current_scenario:
            self._apply_scenario(current_scenario)
            
        # 1. HARDWARE/OS MANAGEMENT
        self.sdv_module.adjust_compute_allocation(self.state, self.model)
        
        # 2. SENSE (Real Hardware)
        self.model.detected_objects = self.perception.process_sensors()
        self.dynamics_reader.update_state_from_feedback(self.state, self.profile) # Get real speed/GPS
        
        # 3. NAVIGATE (Real APIs)
        self.model.current_law = self.navigator.get_current_law(self.state)
        nav_prompt = self.navigator.get_route_prompt_segment(self.state)
        
        # NEW: Check for mission completion
        if "Arrived" in self.model.current_law.upcoming_maneuver:
            print("[Core] Destination reached. Disengaging autonomous drive.")
            self.mission_active = False
            # Send a final "park" command
            self.state.target_acceleration_request = 0.0
            self.state.target_steering_request = 0.0
            self.control.send_safe_park_command(self.profile)
            return # Stop the cycle here

        # 4. THINK (Real AI)
        self.planning.get_cognitive_plan(self.state, self.model, nav_prompt)
        
        # 5. VALIDATE (Cyber Security)
        self.cyber_security.check_for_anomalies(self.state)
        self.cyber_security.validate_command(self.state, self.profile) # Clamps commands if unsafe
        
        # 6. ACT (Real Hardware)
        # Update final commands *after* validation
        self.state.acceleration_command = self.state.target_acceleration_request
        self.state.steering_command = self.state.target_steering_request
        self.control.execute_commands(self.state, self.profile)

        # 7. SUMMARY
        print("--- SUMMARY ---")
        print(f"L5 Status: {self.state.current_planning_state} | Speed: {self.state.speed_mps:.2f} m/s")
        print(f"GPS: ({self.state.gps[0]:.4f}, {self.state.gps[1]:.4f}) | Law: {self.model.current_law.max_speed_limit_mps:.1f} m/s")
        print(f"Next Turn: {self.model.current_law.upcoming_maneuver}")
            
    def run_main_loop(self):
        """
        NEW: This is the persistent service loop for the "installable" program.
        It runs forever, waiting for missions.
        """
        print("[Core] Main autonomous service loop started. Awaiting mission...")
        while True:
            try:
                if self.mission_active:
                    # --- DRIVE CYCLE ---
                    self.run_cycle()
                    # Run at the real-world cycle time
                    time.sleep(0.4) 
                else:
                    # --- STANDBY CYCLE ---
                    # No mission, just idle.
                    # We can run checks at a much lower frequency.
                    self.state.current_planning_state = "STANDBY"
                    # Run a low-power self-check
                    self.cyber_security.check_for_anomalies(self.state)
                    time.sleep(1.0) # Wait 1 second before checking for a mission again

            except Exception as e:
                # --- CRITICAL ERROR HANDLING ---
                print(f"[Core] FATAL ERROR in main loop: {e}")
                print("[Core] Attempting to enter SAFE PARK mode.")
                self.mission_active = False
                self.state.current_planning_state = "ERROR_STATE"
                try:
                    self.control.send_safe_park_command(self.profile)
                except Exception as e_can:
                    print(f"[Core] FATAL: CAN bus failed during safe park attempt: {e_can}")
                
                print("[Core] System halted. Requires manual restart.")
                break # Exit the loop


# --- 7. Agile Factory / Main Execution ---
# This is where the production system is assembled and installed.

if __name__ == '__main__':
    
    print("--- [AGILE BUILD] Assembling L5 Production Service (Zero-Hardware Profile) ---")
    
    # 1. Define the Vehicle Profile (No LiDAR)
    prod_profile = VehicleProfile(
        name="L5 Sedan (Camera/Radar Only)",
        has_lidar=False
    )
    
    # 2. Define API Keys (Must be provisioned on the device)
    GEMINI_API_KEY = "" # Provisioned by the Android OS
    MAPS_API_KEY = ""   # Provisioned by the Android OS

    # 3. Instantiate State and Model
    model = EnvironmentModel()
    
    # 4. Instantiate Concrete Production Modules
    planner = GeminiCognitivePlanner(GEMINI_API_KEY)
    perception = AndroidSensorSuite(prod_profile)
    can_interface = AndroidCANInterface(prod_profile)
    sdv = AndroidComputeModule()
    cyber = CyberSecurityModule(model)
    learning = SelfLearningModule()
    navigator = GoogleMapsNavigator(MAPS_API_KEY)
    
    # 5. Inject Dependencies into the Orchestrator
    system = GlobalAutonomySystem(
        vehicle_profile=prod_profile,
        perception=perception,
        planner=planner,
        control=can_interface,       # The CAN interface IS the controller
        dynamics_reader=can_interface, # The CAN interface IS the dynamics reader
        sdv_module=sdv,
        cyber_security=cyber,
        self_learning=learning,
        navigator=navigator
    )
    
    # 6. Link cross-module dependencies
    system.planning.self_learning_module = learning
    
    # 7. Start the main service loop
    # This loop will run forever, waiting for a mission.
    # In a real app, this would be started by the Android OS.
    
    # --- SIMULATING THE "APP" ---
    # We will start the main loop, then simulate a user
    # entering a destination in the Android UI 5 seconds later.
    
    # This would be in a separate thread in a real app, but
    # for this example, we'll just show the concept.
    
    # system.run_main_loop() 
    
    # --- Example of how the system would be used ---
    print("\n--- [Service Demo] ---")
    print("System is now in STANDBY mode, waiting for mission...")
    print("... (Simulating 3 seconds of standby) ...")
    time.sleep(3)
    
    # This is the call that would come from the Android UI
    # when the user types "Disneyland" and hits "Go".
    system.set_mission("Disneyland")
    
    # Now, the main loop (which would be running)
    # will see `self.mission_active == True` and start driving.
    # We'll just run a few cycles to demonstrate.
    for _ in range(10):
        if system.mission_active:
            system.run_cycle()
            time.sleep(0.4)
        else:
            print("[Core] Mission complete. Returning to STANDBY.")
            break
