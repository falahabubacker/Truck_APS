import numpy as np
import gymnasium as gym
import carla
import queue
from collections import deque
import time
import os
# import open3d as o3d
# from matplotlib import cm
import math

NUM_RADARS = 10              # 3 on truck, 7 on trailer
RADAR_RANGE = 4          # Max range of radars in meters
MAX_POINTS_PER_SENSOR = 1  # Max detections to store per sensor

def get_actor_pose(actor):
    """Gets an actor's [x, y, yaw] pose as a numpy array."""
    if not actor:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    try:
        t = actor.get_transform()
    except Exception as e:
        t = actor.transform
    return np.array([
        t.location.x,
        t.location.y,
        t.rotation.yaw
    ], dtype=np.float32)

def find_obj(world, keyword, offset=0):
    env_objs = world.get_environment_objects()
    objs = []

    for env_obj in env_objs:
        if keyword in env_obj.name:
            _env_obj = env_obj
            _env_obj.transform.location.z += offset
            objs.append(_env_obj)

    return objs

class ParkingLotEnv(gym.Env):

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.world.get_spectator().set_transform(carla.Transform(carla.Location(125.234566, 46.220818, 26.722748), carla.Rotation(-74.851135, -90.247307, -0.000359)))

        # region Getting vehicle and sensor blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.truck_bp = self.blueprint_library.find("vehicle.daf.dafxf")
        self.trailer_bp = self.blueprint_library.find("vehicle.trailer.trailer")
        self.collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.radar_bp = self.blueprint_library.find("sensor.other.radar")
        # endregion
        
        # region Getting truck, trailer and parking coordinates/spawn points
        self.truck_spawn_point = find_obj(self.world, "truck_spawn", 0.1)[0]
        self.trailer_spawn_point = find_obj(self.world, "trailer_spawn", 0.1)[0]
        self.truck_parking_point = find_obj(self.world, "truck_parking")[0]
        self.parking_point = find_obj(self.world, "trailer_parking")[0]
        # endregion

        self.env_boundary = find_obj(self.world, "env_boundary")[0]
        self.env_boundary_vertices = [self.env_boundary.bounding_box.get_world_vertices(self.env_boundary.transform)[vertex] for vertex in range(0, 8, 2)]
        
        # region Action and Observation space
        # Setting observation space [truck_pose, trailer_pose, parking_pose(trailer), truck_parking_pose, current_stage, 
        #                            distance_to_target, angle_difference, jackknife_angle, phi(angle b/w truck n trailer)]
        self.observation_space = gym.spaces.Dict(
            {                
                # Current parking stage: 0 = positioning (go to truck_parking), 1 = backing (go to trailer_parking)
                "current_stage": gym.spaces.Discrete(2),
                
                # Engineered Features for Better RL Performance
                # Distance from trailer to parking spot (meters)
                "distance_to_target": gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
                
                # Angular difference between trailer and target orientation (degrees, -180 to 180)
                "angle_difference": gym.spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                
                # Jackknife angle between truck and trailer (degrees, -180 to 180)
                "jackknife_angle": gym.spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                
                # Angle from trailer back end to parking spot (degrees, -180 to 180)
                "phi": gym.spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float32),
                
                # Lateral distance perpendicular to parking spot orientation (meters)
                "parallel_distance": gym.spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32),
                
                # Longitudinal distance along parking spot forward direction (meters)
                "longitudinal_distance": gym.spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32),
                
                # Radar data: [NUM_RADARS, MAX_POINTS_PER_SENSOR]
                # Each value is the 'depth' (distance) of a detection
                "radar_data": gym.spaces.Box(
                    low=0, 
                    high=RADAR_RANGE, 
                    shape=(NUM_RADARS, 1), 
                    dtype=np.float32
                )
            }
        )

        # Action space [steering and throttle]
        self.action_space = gym.spaces.Box(low=np.array([-0.8, 0.04]), # Steering, Throttle (reduced to min)
                                            high=np.array([0.8, 0.05]))  # Max throttle reduced to 0.02
        # endregion

        # region Constants and radar settings 
        # Class state variables
        self.actor_list = []
        self.episode_number = 0  # Counter to track episodes
        self.episode_offset = 0  # Offset when resuming from checkpoint
        
        # Stage tracking for 2-stage parking
        self.current_stage = 0  # 0 = positioning, 1 = backing
        self.stage_1_completed = False
        
        # Stage transition thresholds
        self.stage_1_distance_threshold = 1.0  # meters
        self.stage_1_angle_threshold = 15.0    # degrees

        # Reward calculation parameters
        self.min_distance = 1.0  # Minimum distance threshold (meters)
        self.max_distance = 30.0  # Maximum distance threshold (meters)
        
        # Sensor data handling
        self.radar_queues = [queue.Queue() for _ in range(NUM_RADARS)]
        self.collision_truck_history = []
        self.collision_trailer_history = []
        
        # Configure the radar blueprint
        self.radar_bp.set_attribute('horizontal_fov', '60')
        self.radar_bp.set_attribute('vertical_fov', '15')
        self.radar_bp.set_attribute('range', f'{RADAR_RANGE}')

        self.radar_data = None
        # endregion

        self.prev_distance = None
        self.prev_angle_difference = None
        self.prev_jackknife_angle = None

    def reset(self, seed=None, options=None):

        super().reset(seed=None)
        
        # Increment episode counter
        self.episode_number += 1

        # Destroy all actors
        self.destroy_actors()
        self.collision_truck_history.clear()
        self.collision_trailer_history.clear()
        for q in self.radar_queues:
            while not q.empty():
                q.get()
        
        # Reset stage tracking
        self.current_stage = 0
        self.stage_1_completed = False
        self._stage_1_bonus_given = False  # Reset bonus flag for new episode
        
        # region Spawn Truck, trailer, attach them and attach sensors and get first reading
        spawn_transform = self.truck_spawn_point.transform
        self.truck = self.world.try_spawn_actor(self.truck_bp, spawn_transform)
        self.actor_list.append(self.truck)
        print("Truck Spawned.")

        # Spawn Trailer and Attach
        trailer_spawn_base = self.truck_spawn_point.transform
        new_trailer_location = trailer_spawn_base.location - trailer_spawn_base.get_forward_vector() * 4.0
        trailer_transform = carla.Transform(new_trailer_location, trailer_spawn_base.rotation) # trailer_transform.location.z += 0.2 

        self.trailer = self.world.try_spawn_actor(self.trailer_bp, trailer_transform) 
        self.actor_list.append(self.trailer)
        print("Trailer Spawned and Attached.")
        
        physics_control = self.truck.get_physics_control()
        physics_control.max_rpm = 2000.0  # Lower = less power
        self.truck.apply_physics_control(physics_control)
        
        self.world.tick()
        time.sleep(0.1)

        # Spawn and Attach Sensors
        self.spawn_sensors()
        
        # Wait for the first radar data to arrive from all sensors
        for i, q in enumerate(self.radar_queues):
            while q.empty():
                print(f"Waiting for radar sensor {i}...")
                time.sleep(0.01)
        
        # endregion
        
        self.strt = time.perf_counter()

        obs = self._get_observation()

        self.prev_distance = float(obs["distance_to_target"][0])
        self.prev_angle_difference = abs(float(obs["angle_difference"][0]))
        self.prev_jackknife_angle = float(obs["jackknife_angle"][0])
        
        return obs, {}
    
    def spawn_sensors(self):
        """Helper function to create and attach all sensors."""
        
        # Collision Sensors
        collision_tf = carla.Transform() # Attach at vehicle origin
        
        col_sensor_truck = self.world.spawn_actor(self.collision_bp, collision_tf, attach_to=self.truck)
        col_sensor_truck.listen(self.collision_truck_history.append)
        self.actor_list.append(col_sensor_truck)
        
        col_sensor_trailer = self.world.spawn_actor(self.collision_bp, collision_tf, attach_to=self.trailer)
        col_sensor_trailer.listen(lambda event: self.collision_trailer_history.append(event) if event.other_actor.type_id != 'static.unknown' else None)
        self.actor_list.append(col_sensor_trailer)

        truck_bb = self.truck.bounding_box
        trailer_bb = self.trailer.bounding_box
        
        # Define Sensor Mount Points
        # (x, y, z, yaw) relative to the vehicle
        self.truck_radar_mounts = [
            (truck_bb.extent.x - 0.75, 0.0, 0.5, 0.0),   # Front
            (0.0, truck_bb.extent.y - 0.34, 0.5, 90.0),   # Right
            (0.0, -truck_bb.extent.y + 0.34, 0.5, -90.0)  # Left
        ]
        self.trailer_radar_mounts = [
            (-trailer_bb.extent.x * 2.1, 0.0, 0.5, 180.0), # Back Center
            (-trailer_bb.extent.x * 0.7, trailer_bb.extent.y + 0.2, 0.5, 90.0),   # Right Front
            (-trailer_bb.extent.x * 0.7, -trailer_bb.extent.y - 0.2, 0.5, -90.0),  # Left Front
            (-trailer_bb.extent.x * 1.8, trailer_bb.extent.y + 0.2, 0.5, 90.0),   # Right Back
            (-trailer_bb.extent.x * 1.8, -trailer_bb.extent.y - 0.2, 0.5, -90.0), # Left Back
            (-trailer_bb.extent.x * 2.1, trailer_bb.extent.y + 0.0, 0.5, 180.0),  # Right Back Corner
            (-trailer_bb.extent.x * 2.1, -trailer_bb.extent.y - 0.0, 0.5, 180.0)  # Left Back Corner
        ]

        # Radar Sensors
        # 3 on the truck
        for i, (x, y, z, yaw) in enumerate(self.truck_radar_mounts):
            tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
            sensor = self.world.spawn_actor(self.radar_bp, tf, attach_to=self.truck)
            # sensor.listen(self.radar_queues[i].put)
            sensor.listen(lambda radar_data, queue_id=i: self.radar_callback(radar_data, queue_id))
            self.actor_list.append(sensor)
            
        # 7 on the trailer
        for i, (x, y, z, yaw) in enumerate(self.trailer_radar_mounts):
            j = i + 3 # Offset index for queues and list
            tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
            sensor = self.world.spawn_actor(self.radar_bp, tf, attach_to=self.trailer)
            # sensor.listen(self.radar_queues[j].put)
            sensor.listen(lambda radar_data, queue_id=j: self.radar_callback(radar_data, queue_id))
            self.actor_list.append(sensor)
            
        print(f"Spawned {len(self.actor_list)} actors (vehicles + sensors).")

    def _get_observation(self):
        """Helper function to assemble the observation dictionary."""
        # Get Pose Data
        truck_pose = get_actor_pose(self.truck)
        trailer_pose = get_actor_pose(self.trailer)
        parking_pose = get_actor_pose(self.parking_point)
        truck_parking_pose = get_actor_pose(self.truck_parking_point)
        
        actor_pose = trailer_pose  # For engineered features, we consider the trailer as the main actor
        # Determine current target based on stage
        if self.current_stage == 0:
            # Stage 1: Target is truck_parking_point, measure from truck
            target_pose = truck_parking_pose
        else:
            # Stage 2: Target is trailer_parking_point, measure from trailer
            target_pose = parking_pose
        
        # Extract components
        trailer_x, trailer_y, trailer_yaw = trailer_pose
        truck_x, truck_y, truck_yaw = truck_pose
        target_x, target_y, target_yaw = target_pose
        actor_x, actor_y, actor_yaw = actor_pose
        
        # ===== Compute Engineered Features Based on Current Stage =====
        # FIX: In Stage 1, measure from TRUCK (since truck_parking is the truck's target)
        # In Stage 2, measure from TRAILER (since trailer_parking is the trailer's target)
        if self.current_stage == 0:
            # Stage 1: Measure from truck's front
            truck_yaw_rad = np.deg2rad(truck_yaw)
            truck_forward = np.array([np.cos(truck_yaw_rad), np.sin(truck_yaw_rad)])
            truck_bb = self.truck.bounding_box
            reference_point = np.array([truck_x, truck_y]) + truck_forward * truck_bb.extent.x
        else:
            # Stage 2: Measure from trailer's back end
            trailer_yaw_rad = np.deg2rad(trailer_yaw)
            trailer_backward = np.array([-np.cos(trailer_yaw_rad), -np.sin(trailer_yaw_rad)])
            trailer_bb = self.trailer.bounding_box
            reference_point = np.array([trailer_x, trailer_y]) + trailer_backward * trailer_bb.extent.x * 2.1
        
        # 1. Distance to target
        distance_to_target = np.linalg.norm(reference_point - target_pose[:2])
        
        # 2. Angle difference (actor orientation vs target orientation)
        # IMPORTANT: Use the correct vehicle's orientation for the current stage
        if self.current_stage == 0:
            # Stage 0: truck should align with truck_parking
            actor_yaw_for_angle = truck_yaw
        else:
            # Stage 1: trailer should align with trailer_parking
            actor_yaw_for_angle = trailer_yaw
        
        angle_diff = (actor_yaw_for_angle - target_yaw + 180) % 360 - 180
        
        # 3. Jackknife angle (truck orientation vs trailer orientation)
        jackknife = (truck_yaw - trailer_yaw + 180) % 360 - 180
        
        # 4. Phi: Angle from reference point to target (relative to actor's backward direction)
        vec_to_target = np.array([target_pose[0] - reference_point[0], target_pose[1] - reference_point[1]])
        world_angle_to_target = np.rad2deg(np.arctan2(vec_to_target[1], vec_to_target[0]))
        phi = (world_angle_to_target - actor_yaw + 180) % 360 - 180
        
        # 5. Position and distance calculations relative to target
        target_yaw_rad = np.deg2rad(target_yaw)
        target_forward = np.array([np.cos(target_yaw_rad), np.sin(target_yaw_rad)])
        
        # Vector from target to actor
        to_actor = np.array([actor_x - target_x, actor_y - target_y])
        
        # Longitudinal distance (along target's forward direction)
        longitudinal_dist = np.dot(to_actor, target_forward)
        
        # Parallel distance (perpendicular to target's forward direction)
        target_right = np.array([-target_forward[1], target_forward[0]])
        parallel_dist = np.dot(to_actor, target_right)
        
        # Process Radar Data
        # Initialize radar data array, padding with max range
        radar_obs = np.full((NUM_RADARS, MAX_POINTS_PER_SENSOR), RADAR_RANGE, dtype=np.float32)
        
        for i in range(NUM_RADARS):
            # Get the latest data packet from this sensor's queue

            while not self.radar_queues[i].empty():
                try:
                    self.radar_data = self.radar_queues[i].get_nowait()
                except queue.Empty:
                    break
            
            # Get 'depth' (distance) for each detection
            detections = np.array([d.depth for d in self.radar_data])
            
            # Clamp detections to max range
            detections[detections > RADAR_RANGE] = RADAR_RANGE
            
            # Sort by distance (closest first)
            detections = np.sort(detections)
            
            # Fill the observation array with the closest detections
            num_detections = min(len(detections), MAX_POINTS_PER_SENSOR)
            if num_detections > 0:
                radar_obs[i, 0] = detections[0]

        return {
            "current_stage": self.current_stage,
            "distance_to_target": np.array([distance_to_target], dtype=np.float32),
            "angle_difference": np.array([angle_diff], dtype=np.float32),
            "jackknife_angle": np.array([jackknife], dtype=np.float32),
            "phi": np.array([phi], dtype=np.float32),
            "parallel_distance": np.array([parallel_dist], dtype=np.float32),
            "longitudinal_distance": np.array([longitudinal_dist], dtype=np.float32),
            "radar_data": radar_obs
        }
    
    def step(self, action):
        """
        Applies an action, ticks the world, and returns the next (obs, reward, done, info).
        """
        obs = self._get_observation()
        
        self.prev_distance = float(obs["distance_to_target"][0])
        self.prev_angle_difference = abs(float(obs["angle_difference"][0]))
        self.prev_jackknife_angle = float(obs["jackknife_angle"][0])

        # 1. Apply action to self.truck
        steer = float(action[0])
        throttle = float(action[1]) / 1.5 # Removed /1.5 division that was making it too slow
        
        # IMPORTANT: When reversing, steering is inverted in CARLA
        # Negate steering so that negative action = left turn, positive = right turn (consistent with forward driving)
        steer_reversed = -steer  # Invert steering for reverse
        
        # Apply control with "fixed in reverse"
        self.truck.apply_control(carla.VehicleControl(
            steer=steer_reversed, 
            throttle=throttle, 
            # brake=brake, 
            reverse=True
        ))
        
        # 2. Tick the world to advance simulation
        self.world.tick()
        
        # 3. Get the new observation
        obs = self._get_observation()
        
        # 3.5 Check for stage transition (Stage 1 -> Stage 2)
        if self.current_stage == 0 and not self.stage_1_completed:
            trailer_pose = get_actor_pose(self.trailer)
            truck_parking_pose = get_actor_pose(self.truck_parking_point)
            
            # Calculate distance from trailer to truck_parking
            dist_to_truck_parking = np.linalg.norm(trailer_pose[:2] - truck_parking_pose[:2])
            
            # Calculate angle difference
            trailer_yaw = trailer_pose[2]
            truck_parking_yaw = truck_parking_pose[2]
            yaw_diff = abs((trailer_yaw - truck_parking_yaw + 180) % 360 - 180)
            
            # Check if trailer reached the positioning point
            if dist_to_truck_parking < self.stage_1_distance_threshold and yaw_diff < self.stage_1_angle_threshold:
                self.current_stage = 1
                self.stage_1_completed = True
                print("\n*** STAGE 1 COMPLETE! Transitioning to Stage 2: Backing into Parking ***\n")
                
                # CRITICAL FIX: Reset previous state values when transitioning stages
                # Otherwise, delta calculation compares old reference point (truck) with new reference point (trailer)
                # which causes corrupted reward signals at the transition
                self.prev_distance = float(obs["distance_to_target"][0])
                self.prev_angle_difference = abs(float(obs["angle_difference"][0]))
                self.prev_jackknife_angle = float(obs["jackknife_angle"][0])
        
        # 4. Calculate reward
        reward = float(self._calculate_reward(obs))
        
        # Display metrics as HUD (fixed position relative to spectator camera)
        spectator = self.world.get_spectator()
        spectator_transform = spectator.get_transform()
        
        # Position text in front and to the top-left of camera view
        forward = spectator_transform.get_forward_vector()
        right = spectator_transform.get_right_vector()
        up = spectator_transform.get_up_vector()
        
        # Calculate HUD position (5m forward, 2m left, 2m up from camera)
        hud_location = spectator_transform.location + forward * 5.0 - right * 2.0 + up * 2.0
        
        distance_to_target = float(obs["distance_to_target"][0])
        phi = float(obs["phi"][0])
        jackknife = float(obs["jackknife_angle"][0])
        angle_diff_val = float(obs["angle_difference"][0])
        stage_name = "Stage 1: Positioning" if self.current_stage == 0 else "Stage 2: Backing"
        
        text = f"{stage_name}\nDistance: {distance_to_target:.2f}m\nAngle diff: {angle_diff_val:.1f}°\n\
        Phi: {phi:.1f}°\nJackknife: {jackknife:.1f}°\nReward: {reward:.3f}."
        self.world.debug.draw_string(
            hud_location,
            text,
            draw_shadow=True,
            color=carla.Color(r=0, g=50, b=0) if self.current_stage == 0 else carla.Color(r=255, g=165, b=0),
            life_time=0.05,
        )

        terminated = False
        truncated = False
        info = {}

        # Check for Crash (Terminated)
        if len(self.collision_truck_history) > 0 or len(self.collision_trailer_history) > 0:
            terminated = True
            reward = -10.0  # Collision penalty
            print(f"Collision detected! {self.collision_trailer_history[-1].other_actor if len(self.collision_trailer_history) > 0 else self.collision_truck_history[-1].other_actor}")
        
        # Check how long the simulation has been running
        if time.perf_counter() - self.strt > 150:
            terminated = True
            reward = -10.0  # Timeout penalty
            print("Simulation run for too long!")

        # Check for success (only in Stage 2)
        if self.current_stage == 1:
            trailer_pose_check = get_actor_pose(self.trailer)
            parking_pose_check = get_actor_pose(self.parking_point)
            dist_to_target = np.linalg.norm(trailer_pose_check[:2] - parking_pose_check[:2])
            
            trailer_yaw = trailer_pose_check[2]
            parking_yaw = parking_pose_check[2]
            yaw_diff = (trailer_yaw - parking_yaw + 180) % 360 - 180
            
            # If we are within 1 meter and 10 degrees, we win!
            if dist_to_target < 1.0 and abs(yaw_diff) < 10.0:
                terminated = True
                reward = 10.0  # Big positive reward for success!
                print("\n=== PARKING COMPLETE! ===\n")
        
        # 6. Info dict (optional)
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_jackknife_penalty(self, jackknife_angle):
        abs_jackknife = abs(jackknife_angle)
        
        # Soft start: no penalty for small angles
        if abs_jackknife < 20:
            return 0.0
        
        # Progressive penalty for moderate jackknife (20-60°)
        if abs_jackknife < 60:
            # Linear growth: starts at 0 at 20°, reaches ~0.2 at 60°
            penalty = (abs_jackknife - 20) * 0.005
            return penalty
        
        # Heavy penalty for severe jackknife (>60°)
        # Exponential growth: 0.2 at 60°, 0.5 at 70°, 1.0 at 80°+
        penalty = 0.2 + (abs_jackknife - 60) * 0.04
        
        return min(penalty, 1.0)  # Cap at 1.0 to prevent explosions

    def _calculate_reward(self, obs):
        # Extract current state scalars
        distance = float(obs["distance_to_target"][0])
        angle_diff = abs(float(obs["angle_difference"][0]))
        jackknife = float(obs["jackknife_angle"][0])

        # Calculate deltas (improvement)
        distance_delta = self.prev_distance - distance
        angle_delta = abs(self.prev_angle_difference) - abs(angle_diff)
        
        # === 1. Delta-Based Rewards (improvement) ===
        # Reward improvement, but don't over-penalize necessary detours
        angle_improvement = 20 * angle_delta  # Reduced from 35
        distance_improvement = 30 * distance_delta  # Reduced from 65
        
        # Symmetric scaling (removed 1.5x penalty bias)
        # Small asymmetry is OK, but 1.5x was too harsh
        angle_improvement *= 1.2 if angle_improvement < 0 else 1.0
        distance_improvement *= 1.2 if distance_improvement < 0 else 1.0
        
        # === 2. State-Based Rewards (being in good position) ===
        # Reward for being close to target (exponential decay)
        # At distance=1m: +1.0, at distance=5m: +0.2, at distance=10m: +0.05
        proximity_reward = np.exp(-distance / 5.0)
        
        # Reward for good alignment (but clip to prevent negative penalties from dominating)
        # At 0°: +1.0, at 45°: +0.5, at 90°: 0.0, at 180°: -0.3 (clipped to prevent collapse)
        alignment_raw = np.cos(np.deg2rad(angle_diff))
        alignment_reward = np.clip(alignment_raw, -0.3, 1.0)  # Clamp negative to -0.3 max
        
        # === 3. Jackknife Penalty ===
        jackknife_penalty = self._calculate_jackknife_penalty(jackknife)

        # === 4. Stage Completion Bonus ===
        stage_bonus = 0.0
        if self.current_stage == 1 and not self._stage_1_bonus_given:
            stage_bonus = 100  # Increased for better signal
            self._stage_1_bonus_given = True
        
        # === 5. Small time penalty to encourage efficiency ===
        time_penalty = 0.01

        # === Combined Reward ===
        # Balance improvement (delta) with baseline state rewards
        total_reward = (
            angle_improvement +           # Reward for improving angle
            distance_improvement +        # Reward for getting closer
            proximity_reward * 5.0 +     # Reward for being close (baseline)
            alignment_reward * 3.0 +     # Reward for being aligned (baseline)
            stage_bonus -                # Big bonus for stage completion
            jackknife_penalty -          # Penalty for jackknifing
            time_penalty                 # Small efficiency penalty
        )

        return total_reward
    
    def destroy_actors(self):
        """Helper function to destroy all spawned actors."""
        print(f"Destroying {len(self.actor_list)} actors...")
        for actor in self.actor_list:
            if actor and actor.is_alive:
                # Sensors must be stopped before destroying
                if 'sensor' in actor.type_id:
                    actor.stop()
                actor.destroy()
        self.actor_list.clear()
    
    def close(self):
        """
        Closes the environment and destroys all actors.
        """
        self.destroy_actors()
        # --- FIX 2: Add a small delay to prevent sensor warnings ---
        time.sleep(2.0)
    
    def radar_callback(self, radar_data, queue_id):

        try:
            self.radar_queues[queue_id].put(radar_data)
            # region draw arrow and radar signal
            # current_rot = radar_data.transform.rotation
            # for detect in radar_data:
            #     azi = math.degrees(detect.azimuth)
            #     alt = math.degrees(detect.altitude)
            #     fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            #     carla.Transform(
            #         carla.Location(),
            #         carla.Rotation(
            #             pitch=current_rot.pitch + alt,
            #             yaw=current_rot.yaw + azi,
            #             roll=current_rot.roll)
            #         ).transform(fw_vec)

            #     norm_velocity = detect.velocity / 7.5 # range [-1, 1]
            #     r = 255 # int(max(0.0, min(1.0, 1.0 - norm_velocity)) * 255.0)
            #     g = 0 # int(max(0.0, min(1.0, 1.0 - abs(norm_velocity))) * 255.0)
            #     b = 0 # int(abs(max(- 1.0, min(0.0, - 1.0 - norm_velocity))) * 255.0)
                
            #     self.world.debug.draw_point(radar_data.transform.location + fw_vec, size=0.1, color=carla.Color(r, g, b), life_time=0.05)

            #     for sensor in self.world.get_actors().filter("sensor.other.radar*"):
            #         sensor_transform = sensor.get_transform()

            #         self.world.debug.draw_arrow(
            #             sensor_transform.location,
            #             sensor_transform.location + sensor_transform.get_forward_vector() * 1.3,
            #             thickness=0.1,
            #             arrow_size=0.1,
            #             color=carla.Color(r=255, g=125, b=0),
            #             life_time=0.01
            #         )
            # endregion

        except Exception as e:
            print(f"Error in radar callback: {e}")

    def set_episode_offset(self, offset):
        """Set the episode number offset when resuming from checkpoint."""
        self.episode_offset = offset
        self.episode_number = offset
    
    def get_episode_state(self):
        """Get the current episode state for saving to checkpoint."""
        return {
            'episode_number': self.episode_number,
            'episode_offset': self.episode_offset
        }
    
    def restore_episode_state(self, state):
        """Restore episode state from saved checkpoint."""
        if state:
            self.episode_number = state.get('episode_number', 0)
            self.episode_offset = state.get('episode_offset', 0)