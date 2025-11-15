import random
import time

# --- SAE Level Constants ---
SAE_LEVEL_2 = "L2_Partial_Automation (TSS 3.0)"
SAE_LEVEL_0 = "L0_Human_Driving"

# --- System State Enums ---
class SystemState:
    """Defines the operational state of the L2 system."""
    SYSTEM_OFF = "System Off (Human Driving)"
    LTA_ACC_ACTIVE = "LTA & ACC Active (High-Speed, Hands-On)"
    TRAFFIC_JAM_ASSIST = "Traffic Jam Assist Active (Low-Speed, Hands-Free)"
    LANE_CHANGE_ASSIST = "Lane Change Assist (Executing Maneuver)"
    DRIVER_WARNING_STAGE_1 = "Stage 1 Warning (Visual/Audible)"
    DRIVER_WARNING_STAGE_2 = "Stage 2 Warning (Haptic/Brake Taps)"
    SYSTEM_DISENGAGED = "System Disengaged (ODD Breach / Driver Unresponsive)"
    EMERGENCY_STOP = "Emergency Driving Stop System (EDSS)"
    CRITICAL_SAFETY_BUFFER = "Critical Safety Buffer (L3-Style Soft Fallback)"

class VehicleConfiguration:
    """Defines model-specific operational parameters."""
    def __init__(self, model_year: int):
        self.model_year = model_year
        if model_year == 2024:
            self.HANDS_FREE_MAX_SPEED = 40  # 2024: Hands-free only in TJA below 40 km/h
            self.CRITICAL_BUFFER_DURATION = 5 # Standard buffer time
            self.LCA_DEGRADED_MODE_ENABLED = True # LCA disabled in Light Rain
        elif model_year == 2025:
            self.HANDS_FREE_MAX_SPEED = 60  # 2025: Hands-free expanded up to 60 km/h (Enhanced TJA/Cruising)
            self.CRITICAL_BUFFER_DURATION = 7 # Enhanced sensor fusion offers longer buffer
            self.LCA_DEGRADED_MODE_ENABLED = False # LCA enabled in Light Rain (due to better sensors)
            
        print(f"| CONFIG: Loaded {model_year} parameters. Hands-Free Max: {self.HANDS_FREE_MAX_SPEED} km/h.")


class Environment:
    """Represents the vehicle's current state and external conditions."""
    def __init__(self):
        self.speed_kmh = 70
        self.lane_lines = "Clear"
        self.weather = "Clear"
        self.odd_breached = False # Environmental ODD
        self.is_traffic_jam = False
        self.is_blind_spot_clear = True
        # Proactive Driving Assist (PDA) Hazards
        self.upcoming_hazard = "None" # 'None', 'SharpCurve', 'SlowVehicle', 'PedestrianNearRoad'
        self.hazard_distance = 999
        # Predictive Lane Positioning Hazard
        self.adjacent_lane_hazard = "None" # 'None', 'Truck'
        # Intent Prediction Hazard
        self.adjacent_vehicle_behavior = "Normal" # 'Normal', 'AggressiveCutIn'

    def update_environment(self):
        """Simulates changing environmental conditions."""
        
        # Simulate traffic flow changes
        if self.is_traffic_jam:
            self.speed_kmh = max(0, self.speed_kmh + random.randint(-3, 5))
            if self.speed_kmh > 50: # Slightly higher cutoff for TJA exit
                self.is_traffic_jam = False
        else:
            self.speed_kmh = max(40, self.speed_kmh + random.randint(-5, 5))
            if self.speed_kmh < 40:
                self.is_traffic_jam = True

        # Simulate environmental ODD
        self.weather = random.choices(["Clear", "Light Rain", "Heavy Rain"], [0.7, 0.2, 0.1])[0]
        self.lane_lines = random.choices(["Clear", "Faded", "None"], [0.9, 0.05, 0.05])[0]
        
        # Determine ODD breach based on combined factors
        if self.weather == "Heavy Rain" or self.lane_lines == "None":
            self.odd_breached = True
        else:
            self.odd_breached = False
            
        # Simulate blind spot
        self.is_blind_spot_clear = random.random() < 0.8 
        
        # Simulate Proactive Hazards
        if not self.odd_breached and not self.is_traffic_jam and self.upcoming_hazard == "None":
            if random.random() < 0.2:
                self.upcoming_hazard = random.choice(["SharpCurve", "SlowVehicle", "PedestrianNearRoad"])
                self.hazard_distance = 150 # meters
        elif self.upcoming_hazard != "None":
            self.hazard_distance -= self.speed_kmh // 7 # Approximate m/s
            if self.hazard_distance < 10:
                self.upcoming_hazard = "None" 

        # Simulate Adjacent Lane Hazards (for Predictive Positioning)
        if not self.is_traffic_jam and self.adjacent_lane_hazard == "None":
             if random.random() < 0.1:
                self.adjacent_lane_hazard = "Truck"
        elif self.adjacent_lane_hazard != "None":
            if random.random() < 0.15:
                self.adjacent_lane_hazard = "None"

        # Simulate Aggressive Intent for Negotiation
        if self.adjacent_vehicle_behavior == "Normal":
            if random.random() < 0.08:
                self.adjacent_vehicle_behavior = "AggressiveCutIn"
        elif self.adjacent_vehicle_behavior == "AggressiveCutIn":
             if random.random() < 0.3:
                self.adjacent_vehicle_behavior = "Normal"

class DriverMonitorCamera:
    """Simulates the Prius's Driver Monitor Camera and driver inputs."""
    def __init__(self):
        self.is_attentive = True # Eyes on road
        self.is_hands_on = True  # Torque on wheel

    def check_driver_inputs(self):
        """Simulates checking all driver states for this loop cycle."""
        # 90% chance of being attentive, 80% chance of hands-on
        self.is_attentive = random.random() < 0.90 
        self.is_hands_on = random.random() < 0.80 
        self.is_turn_signal_on = random.random() < 0.05

class PriusTSS3_Enhanced:
    """The main L2+++ decision-making unit, now modular by model year."""
    def __init__(self, env: Environment, config: VehicleConfiguration):
        self.env = env
        self.config = config
        self.dms = DriverMonitorCamera()
        self.current_state = SystemState.LTA_ACC_ACTIVE
        self.warning_stage_timer = 0
        self.performance_mode = "Optimal" # 'Optimal', 'Degraded'
        self.lane_position = "Center" # 'Center', 'OffsetLeft', 'OffsetRight'
        print(f"--- Toyota Safety Sense ({self.config.model_year}) L2+++ Engaged ---")

    # --- Utility Methods (Proactive AI Layers) ---

    def _handle_proactive_driving_assist(self):
        """Proactive Driving Assist (PDA) layer."""
        if self.env.upcoming_hazard == "SharpCurve" and self.env.hazard_distance < 100:
            print("| PDA: Gently applying brakes for upcoming curve. |")
            self.env.speed_kmh = max(10, self.env.speed_kmh - 2)
                 
        elif self.env.upcoming_hazard == "PedestrianNearRoad" and self.env.hazard_distance < 80:
            print("| PDA: Pedestrian detected! Applying gentle steering bias and braking. |")
            self.env.speed_kmh = max(10, self.env.speed_kmh - 2)

    def _handle_predictive_lane_positioning(self):
        """Lane Offset for adjacent hazards."""
        if self.env.adjacent_lane_hazard == "Truck" and self.lane_position == "Center":
            self.lane_position = "OffsetLeft"
            print(f"| PDA (Offset): Shifting to '{self.lane_position}' to create buffer from truck. |")
        
        elif self.env.adjacent_lane_hazard == "None" and self.lane_position != "Center":
            self.lane_position = "Center"
            
    def _handle_intent_prediction_and_negotiation(self):
        """Responding to aggressive maneuvers."""
        if self.env.adjacent_vehicle_behavior == "AggressiveCutIn":
            print("| AI (Intent): AGGRESSIVE CUT-IN PREDICTED! Applying anticipatory braking. |")
            self.env.speed_kmh = max(10, self.env.speed_kmh - 5)
            if random.random() < 0.5:
                self.env.adjacent_vehicle_behavior = "Normal"

    # --- Core Logic ---

    def execute_driving_loop(self):
        """Main control loop to run the L2+++ system logic."""
        
        self.env.update_environment()
        self.dms.check_driver_inputs()
        
        # Determine the current driver engagement requirement
        is_hands_free_allowed = self.env.speed_kmh <= self.config.HANDS_FREE_MAX_SPEED
        required_engagement = "Eyes-On" if is_hands_free_allowed else "Hands-On/Eyes-On"
        
        # 1. ODD Check
        is_hard_odd_breach = (self.env.weather == "Heavy Rain" and self.env.lane_lines == "None")
        is_soft_odd_breach = self.env.odd_breached and not is_hard_odd_breach

        if is_hard_odd_breach:
            print(f"\n--- ODD HARD BREACH: Vision blocked. ---")
            self.current_state = SystemState.SYSTEM_DISENGAGED
            return
        
        # L3-Style Soft Fallback (Critical Buffer)
        if is_soft_odd_breach and self.current_state not in [SystemState.CRITICAL_SAFETY_BUFFER, SystemState.SYSTEM_DISENGAGED, SystemState.EMERGENCY_STOP]:
             print(f"\n--- ODD SOFT BREACH: Vision degraded. Entering Critical Buffer. ---")
             self.current_state = SystemState.CRITICAL_SAFETY_BUFFER
             self.warning_stage_timer = self.config.CRITICAL_BUFFER_DURATION

        # 2. Dynamic Performance Mode Check (Modular for 2025)
        if self.env.weather == "Light Rain" and self.performance_mode == "Optimal":
            print("| AI: Light Rain detected. Entering 'Degraded' performance mode. |")
            self.performance_mode = "Degraded"
        elif self.env.weather == "Clear" and self.performance_mode == "Degraded":
            self.performance_mode = "Optimal"

        # 3. Call Proactive AI Layers
        if self.current_state in [SystemState.LTA_ACC_ACTIVE, SystemState.TRAFFIC_JAM_ASSIST, SystemState.CRITICAL_SAFETY_BUFFER]:
             self._handle_proactive_driving_assist()
             self._handle_predictive_lane_positioning()
             self._handle_intent_prediction_and_negotiation()

        # 4. Main State Machine
        
        # --- Active Cruising / Hands-Free Cruising ---
        if self.current_state in [SystemState.LTA_ACC_ACTIVE, SystemState.TRAFFIC_JAM_ASSIST]:
            
            is_engaged = self.dms.is_attentive
            if not is_hands_free_allowed:
                is_engaged = is_engaged and self.dms.is_hands_on

            if not is_engaged:
                print(f"\n--- DRIVER ENGAGEMENT LOST ({required_engagement}) ---")
                self.current_state = SystemState.DRIVER_WARNING_STAGE_1
                self.warning_stage_timer = 3
            
            elif self.dms.is_turn_signal_on and self.current_state == SystemState.LTA_ACC_ACTIVE:
                print("\n--- Driver requested lane change. ---")
                self.current_state = SystemState.LANE_CHANGE_ASSIST
            
            # State Transition Logic based on Speed/Config
            elif is_hands_free_allowed:
                if self.current_state == SystemState.LTA_ACC_ACTIVE:
                    print(f"\n--- Speed below {self.config.HANDS_FREE_MAX_SPEED} km/h. Entering Hands-Free Cruising (TJA/Enhanced). ---")
                print(f"| CRUISE: Hands-Free active. Speed: {self.env.speed_kmh} km/h. (Eyes-On) |")
                self.current_state = SystemState.TRAFFIC_JAM_ASSIST # Use TJA state for Hands-Free mode
            
            elif not is_hands_free_allowed:
                 if self.current_state == SystemState.TRAFFIC_JAM_ASSIST:
                    print(f"\n--- Speed above {self.config.HANDS_FREE_MAX_SPEED} km/h. Entering Hands-On Cruising (LTA/ACC). ---")
                    print(">>> Place your hands on the wheel now. <<<")
                 print(f"| CRUISE: Hands-On active. Speed: {self.env.speed_kmh} km/h. ({required_engagement}) |")
                 self.current_state = SystemState.LTA_ACC_ACTIVE

        # --- Lane Change Assist ---
        elif self.current_state == SystemState.LANE_CHANGE_ASSIST:
            
            # LCA Check against Degraded mode
            if self.performance_mode == "Degraded" and self.config.LCA_DEGRADED_MODE_ENABLED:
                print("!!! LCA: CANCELLED. Feature disabled in 'Degraded' (Light Rain) mode. !!!")
                self.current_state = SystemState.LTA_ACC_ACTIVE
                self.dms.is_turn_signal_on = False
                return
                
            if self.env.is_blind_spot_clear:
                print(">>> LCA: Blind spot clear. Executing automated lane change... <<<")
            else:
                print("!!! LCA: BLIND SPOT OCCUPIED! Lane change cancelled. (Beep!) !!!")
            self.current_state = SystemState.LTA_ACC_ACTIVE

        # --- Warning States (1 & 2) ---
        elif self.current_state in [SystemState.DRIVER_WARNING_STAGE_1, SystemState.DRIVER_WARNING_STAGE_2]:
            
            is_re_engaged = self.dms.is_attentive and (self.dms.is_hands_on or is_hands_free_allowed)
            
            if is_re_engaged:
                print("\n--- DRIVER RE-ENGAGED: Warning cleared. ---")
                self.current_state = SystemState.LTA_ACC_ACTIVE if not is_hands_free_allowed else SystemState.TRAFFIC_JAM_ASSIST
                self.warning_stage_timer = 0
            elif self.warning_stage_timer > 0:
                stage = 1 if self.current_state == SystemState.DRIVER_WARNING_STAGE_1 else 2
                print(f"| STAGE {stage} WARNING: Time remaining: {self.warning_stage_timer} cycles. Respond NOW!")
                self.warning_stage_timer -= 1
            elif self.current_state == SystemState.DRIVER_WARNING_STAGE_1:
                print("\n--- STAGE 1 FAILED: Moving to Stage 2 Haptic Warning. ---")
                self.current_state = SystemState.DRIVER_WARNING_STAGE_2
                self.warning_stage_timer = 3
            else: # Stage 2 failed
                print("\n!!! CRITICAL FAILURE: Driver failed to respond. Initiating EDSS!")
                self.current_state = SystemState.EMERGENCY_STOP

        # --- Emergency Stop / Disengaged ---
        elif self.current_state == SystemState.EMERGENCY_STOP:
            print(f"| EDSS: Controlled braking... (Speed: {self.env.speed_kmh})")
            if self.env.speed_kmh > 0:
                self.env.speed_kmh = max(0, self.env.speed_kmh - 5)
                if self.env.speed_kmh <= 0:
                    print("--- EDSS COMPLETE: Vehicle brought to a controlled stop. ---")

        elif self.current_state == SystemState.SYSTEM_DISENGAGED:
            print(f"| TSS: System OFF. {SAE_LEVEL_0} active. Waiting for ODD to be safe.")
            if not self.env.odd_breached:
                print("\n--- ODD Safe: System is available to be re-engaged. ---")
                self.current_state = SystemState.LTA_ACC_ACTIVE
                
        # --- Critical Safety Buffer ---
        elif self.current_state == SystemState.CRITICAL_SAFETY_BUFFER:
            is_re_engaged = self.dms.is_attentive and self.dms.is_hands_on
            
            if not self.env.odd_breached:
                print("\n--- ODD RECOVERED: Exiting Critical Buffer. Resuming LTA/ACC. ---")
                self.current_state = SystemState.LTA_ACC_ACTIVE
                self.warning_stage_timer = 0
            elif is_re_engaged:
                print(f"| CRITICAL BUFFER: Driver re-engagement detected. Disengaging to {SAE_LEVEL_0} (Driver Takeover). |")
                self.current_state = SystemState.SYSTEM_DISENGAGED
            elif self.warning_stage_timer <= 0:
                print("\n!!! CRITICAL BUFFER TIME EXPIRED: Forcing full disengagement. !!!")
                self.current_state = SystemState.SYSTEM_DISENGAGED
            else:
                print(f"| CRITICAL BUFFER: AI maintaining control. Driver MUST re-engage. Timer: {self.warning_stage_timer} cycles.")
                self.warning_stage_timer -= 1
                self.env.speed_kmh = max(20, self.env.speed_kmh - 2) # Slow down for safety


def run_simulation(model_year: int, cycles: int):
    """Runs a dedicated simulation for a specific model year."""
    config = VehicleConfiguration(model_year)
    env = Environment()
    prius_tss = PriusTSS3_Enhanced(env, config)
    
    for cycle in range(1, cycles + 1):
        if prius_tss.current_state == SystemState.EMERGENCY_STOP and prius_tss.env.speed_kmh <= 0:
            print(f"\n--- SIM CYCLE {cycle:02d} | State: {prius_tss.current_state} ---")
            prius_tss.execute_driving_loop()
            print(f"\n--- Simulation for {model_year} ended due to EDSS completion. ---")
            break
            
        print(f"\n--- {model_year} SIM CYCLE {cycle:02d} | State: {prius_tss.current_state} ---")
        prius_tss.execute_driving_loop()
        time.sleep(0.0) # Set to 0.0 for faster simulation output

    print(f"\n--- {model_year} SIMULATION ENDED ---")


def main():
    """Initializes and runs the multi-model simulation."""
    print("=========================================================")
    print("--- Toyota Prius L2+++ Multi-Model Year SIMULATION ---")
    print("=========================================================")

    # Run 2024 Simulation
    run_simulation(2024, 25) # 25 cycles for 2024 model

    print("\n\n=========================================================")
    print("--- Running 2025 Model Simulation (Enhanced ODD) ---")
    print("=========================================================")

    # Run 2025 Simulation
    run_simulation(2025, 25) # 25 cycles for 2025 model

if __name__ == "__main__":
    main()
