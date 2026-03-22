import numpy as np
import gymnasium as gym
import carla
import queue
from collections import deque
import time
import math
import subprocess

NUM_RADARS = 20              # 7 on truck, 13 on trailer
SENSOR_RANGE = 4          # Max range of obstacle sensors in meters
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

def min_max(value, o_min, o_max, n_min=0, n_max=1):
    return (((value - o_min) / (o_max - o_min)) * (n_max - n_min)) + n_min

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def kill_carla():
    subprocess.run(['pkill', '-9', '-f', 'CarlaUE4'], check=False)

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
        self.obstacle_bp = self.blueprint_library.find("sensor.other.obstacle")
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
                "distance_to_target": gym.spaces.Box(low=0.01, high=1.0, shape=(1,), dtype=np.float32),
                
                # Angular difference between trailer and target orientation (degrees, -180 to 180)
                "angle_difference": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                
                # Jackknife angle between truck and trailer (degrees, -180 to 180)
                "jackknife_angle": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                
                # Angle from trailer back end to parking spot (degrees, -180 to 180)
                "phi": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                
                # Lateral distance perpendicular to parking spot orientation (meters)
                "parallel_distance": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                
                # Longitudinal distance along parking spot forward direction (meters)
                "longitudinal_distance": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                
                # Obstacle data: [NUM_RADARS, MAX_POINTS_PER_SENSOR]
                # Each value is the closest detected distance
                "radar_data": gym.spaces.Box(
                    low=0, 
                    high=1.0, 
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
        self.stage_1_distance_threshold = 0.0375  # meters
        self.stage_1_angle_threshold = 0.0555    # degrees

        # Reward calculation parameters
        self.min_distance = 1.0  # Minimum distance threshold (meters)
        self.max_distance = 30.0  # Maximum distance threshold (meters)
        
        # Sensor data handling
        self.obstacle_queues = [queue.Queue() for _ in range(NUM_RADARS)]
        self.collision_truck_history = []
        self.collision_trailer_history = []
        
        # Configure the obstacle blueprint
        self.obstacle_bp.set_attribute('distance', f'{SENSOR_RANGE}')
        self.obstacle_bp.set_attribute('hit_radius', '0.5')
        self.obstacle_bp.set_attribute('only_dynamics', 'false')
        self.obstacle_bp.set_attribute('debug_linetrace', 'false')

        self.obstacle_data = None
        # endregion

        self.init_distance = None

    def _safe_world_tick(self, context="step", timeout=1.0, retries=2):
        """Tick with timeout/retry so training cannot hang forever on stalled frames."""
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                try:
                    self.world.tick(timeout=timeout)
                except TypeError:
                    # Older CARLA builds may not expose timeout argument.
                    self.world.tick()
                return True, None
            except RuntimeError as e:
                last_error = e  
                print(f"world.tick failed in {context} (attempt {attempt}/{retries}): {e}")
                time.sleep(0.1)

        return False, str(last_error)

    def reset(self, seed=None, options=None):

        super().reset(seed=None)
        
        # Increment episode counter
        self.episode_number += 1

        # Destroy all actors
        self.destroy_actors()
        self.collision_truck_history.clear()
        self.collision_trailer_history.clear()
        for q in self.obstacle_queues:
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
        
        tick_ok, tick_error = self._safe_world_tick(context="reset")
        if not tick_ok:
            raise RuntimeError(f"World tick failed during reset: {tick_error}")
        time.sleep(0.1)

        # Spawn and Attach Sensors
        self.spawn_sensors()
        
        # endregion
        
        self.strt = time.perf_counter()

        obs = self._get_observation()

        self.init_distance = float(obs["distance_to_target"][0])
        
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
            (truck_bb.extent.x - 0.45, 0.0, 1, 0.0),   # Front
            (0.0, truck_bb.extent.y - 0.04, 1, 90.0),   # Right
            (truck_bb.extent.x - 0.45, -truck_bb.extent.y + 0.04, 1, -45.0),  # Corner
            (truck_bb.extent.x - 0.45, truck_bb.extent.y - 0.04, 1, 45.0),   # Corner
            (0.0, -truck_bb.extent.y + 0.04, 1, -90.0),  # Left
            (-truck_bb.extent.x * 0.4, truck_bb.extent.y - 0.04, 1, 90.0),   # Mid-Right
            (-truck_bb.extent.x * 0.4, -truck_bb.extent.y + 0.04, 1, -90.0)  # Mid-Left
        ]
        self.trailer_radar_mounts = [
            # Right side (4 sensors)
            (-trailer_bb.extent.x * 0.3, trailer_bb.extent.y + 0.5, 1, 90.0),
            (-trailer_bb.extent.x * 0.9, trailer_bb.extent.y + 0.5, 1, 90.0),
            (-trailer_bb.extent.x * 1.5, trailer_bb.extent.y + 0.5, 1, 90.0),
            (-trailer_bb.extent.x * 2.1, trailer_bb.extent.y + 0.5, 1, 90.0),
            # Left side (4 sensors)
            (-trailer_bb.extent.x * 0.3, -trailer_bb.extent.y - 0.5, 1, -90.0),
            (-trailer_bb.extent.x * 0.9, -trailer_bb.extent.y - 0.5, 1, -90.0),
            (-trailer_bb.extent.x * 1.5, -trailer_bb.extent.y - 0.5, 1, -90.0),
            (-trailer_bb.extent.x * 2.1, -trailer_bb.extent.y - 0.5, 1, -90.0),
            # Back (4 sensors)
            (-trailer_bb.extent.x * 2.1, 0, 1, 180.0),
            (-trailer_bb.extent.x * 2.1, trailer_bb.extent.y * 0.5 + 0.5, 1, 180.0),
            (-trailer_bb.extent.x * 2.1, -trailer_bb.extent.y * 0.5 - 0.5, 1, 180.0),
            (-trailer_bb.extent.x * 2.1, trailer_bb.extent.y * 0.5 + 0.5, 1, 135.0),
            (-trailer_bb.extent.x * 2.1, -trailer_bb.extent.y * 0.5 - 0.5, 1, -135.0),
        ]

        # Obstacle Sensors
        # 7 on the truck
        for i, (x, y, z, yaw) in enumerate(self.truck_radar_mounts):
            tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
            sensor = self.world.spawn_actor(self.obstacle_bp, tf, attach_to=self.truck)
            sensor.listen(lambda obstacle_event, queue_id=i: self.obstacle_callback(obstacle_event, queue_id))
            self.actor_list.append(sensor)
            
        # 13 on the trailer
        for i, (x, y, z, yaw) in enumerate(self.trailer_radar_mounts):
            j = i + len(self.truck_radar_mounts)  # Offset index for queues and list
            tf = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
            sensor = self.world.spawn_actor(self.obstacle_bp, tf, attach_to=self.trailer)
            sensor.listen(lambda obstacle_event, queue_id=j: self.obstacle_callback(obstacle_event, queue_id))
            self.actor_list.append(sensor)
            
        print(f"Spawned {len(self.actor_list)} actors (vehicles + sensors).")

    def _get_observation(self):
        """Helper function to assemble the observation dictionary."""
        # Get Pose Data
        truck_pose = get_actor_pose(self.truck)
        trailer_pose = get_actor_pose(self.trailer)
        truck_parking_pose = get_actor_pose(self.truck_parking_point)
        
        # Extract components
        trailer_x, trailer_y, trailer_yaw = trailer_pose
        truck_x, truck_y, truck_yaw = truck_pose
        target_x, target_y, target_yaw = truck_parking_pose
        
        trailer_yaw_rad = np.deg2rad(trailer_yaw)
        trailer_backward = np.array([-np.cos(trailer_yaw_rad), -np.sin(trailer_yaw_rad)])
        trailer_bb = self.trailer.bounding_box
        reference_point = np.array([trailer_x, trailer_y]) + trailer_backward * trailer_bb.extent.x * 2.1
        
        # 1. Distance to target
        distance_to_target = np.linalg.norm(reference_point - truck_parking_pose[:2]) if not self.stage_1_completed else 0.5
        distance_to_target = min_max(distance_to_target, 0, 40)
        
        angle_diff = (trailer_yaw - target_yaw + 180) % 360 - 180
        angle_diff = min_max(angle_diff, -180, 180)
        
        # 3. Jackknife angle (truck orientation vs trailer orientation)
        jackknife = (truck_yaw - trailer_yaw + 180) % 360 - 180
        jackknife = min_max(jackknife, -180, 180)

        # 4. Phi: Angle from reference point to target (relative to actor's backward direction)
        vec_to_target = np.array([truck_parking_pose[0] - reference_point[0], truck_parking_pose[1] - reference_point[1]])
        world_angle_to_target = np.rad2deg(np.arctan2(vec_to_target[1], vec_to_target[0]))
        phi = (world_angle_to_target - trailer_yaw) % 360 - 180 # Measured wrt backward pointing vector of trailer
        phi = min_max(phi, -180, 180)

        # 5. Position and distance calculations relative to target
        target_yaw_rad = np.deg2rad(target_yaw)
        target_forward = np.array([np.cos(target_yaw_rad), np.sin(target_yaw_rad)])
        
        # Vector from target to actor
        to_actor = np.array([trailer_x - target_x, trailer_y - target_y])
        
        # Longitudinal distance (along target's forward direction)
        longitudinal_dist = np.dot(to_actor, target_forward)
        longitudinal_dist = min_max(longitudinal_dist, 0, 40)
        
        # Parallel distance (perpendicular to target's forward direction)
        target_right = np.array([-target_forward[1], target_forward[0]])
        parallel_dist = np.dot(to_actor, target_right)
        parallel_dist = min_max(parallel_dist, 0, 40)

        # Process obstacle data
        obstacle_obs = np.full((NUM_RADARS, MAX_POINTS_PER_SENSOR), SENSOR_RANGE, dtype=np.float32)
        
        for i in range(NUM_RADARS):
            latest_distance = SENSOR_RANGE
            while not self.obstacle_queues[i].empty():
                try:
                    latest_distance = self.obstacle_queues[i].get_nowait()
                except queue.Empty:
                    break

            latest_distance = clamp(float(latest_distance), 0.0, SENSOR_RANGE)
            obstacle_obs[i, 0] = min_max(latest_distance, SENSOR_RANGE, 0.0)

        return {
            "current_stage": self.current_stage,
            "distance_to_target": np.array([distance_to_target], dtype=np.float32),
            "angle_difference": np.array([angle_diff], dtype=np.float32),
            "jackknife_angle": np.array([jackknife], dtype=np.float32),
            "phi": np.array([phi], dtype=np.float32),
            "parallel_distance": np.array([parallel_dist], dtype=np.float32),
            "longitudinal_distance": np.array([longitudinal_dist], dtype=np.float32),
            "radar_data": obstacle_obs
        }
    
    def step(self, action):
        obs = self._get_observation()
        info = {}
        
        # 1. Apply action to self.truck
        steer = float(action[0])
        throttle = float(action[1]) / 1.4
        
        steer_reversed = steer
        
        # Apply control with "fixed in reverse"
        self.truck.apply_control(carla.VehicleControl(
            steer=steer_reversed, 
            throttle=throttle, 
            # brake=brake, 
            reverse=True
        ))
        
        # 2. Tick the world to advance simulation
        tick_ok, tick_error = self._safe_world_tick(context="step")
        if not tick_ok:
            info["tick_error"] = tick_error
            return obs, -1.0, True, True, info
        
        # 3. Get the new observation
        obs = self._get_observation()
        
        # 3.5 Check for stage transition (Stage 1 -> Stage 2)
        if self.current_stage == 0 and not self.stage_1_completed:
            dist_to_truck_parking = float(obs["distance_to_target"][0])
            
            yaw_diff = float(obs["angle_difference"][0])
            
            # Check if trailer reached the positioning point
            if dist_to_truck_parking < self.stage_1_distance_threshold:
                self.current_stage = 1
                self.stage_1_completed = True
                print("\n*** STAGE 1 COMPLETE! Transitioning to Stage 2: Backing into Parking ***\n")
                
        # 4. Calculate reward
        reward, info = self._calculate_reward(obs)
        
        terminated = False
        truncated = False

        # Check for Crash (Terminated)
        if len(self.collision_truck_history) > 0 or len(self.collision_trailer_history) > 0:
            terminated = True
            reward = -1.0  # Collision penalty
            print(f"Collision detected! {self.collision_trailer_history[-1].other_actor if len(self.collision_trailer_history) > 0 else self.collision_truck_history[-1].other_actor}")
        
        # Check how long the simulation has been running
        if time.perf_counter() - self.strt > 150:
            terminated = True
            reward = -1.0  # Timeout penalty
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
                reward = 1.0  # Big positive reward for success!
                print("\n=== PARKING COMPLETE! ===\n")
        
        # region Display metrics as HUD (fixed position relative to spectator camera)
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
        
        rew_comp = info["reward_comp"]
        text = f"{stage_name}\nDistance: {distance_to_target:.2f}m\nAngle diff: {angle_diff_val:.2f}°\n\
        Phi: {phi:.2f}°\nJackknife: {jackknife:.2f}°\nReward: {reward:.3f}.\n\
        Angle_Rew: {rew_comp['alignment_reward']:.3f}\n\
        Angle_Imp_Rew: {rew_comp['angle_improvement']:.3f}\n\
        Dist_Rew: {rew_comp['proximity_reward']:.3f}\n\
        Dist_Imp_Rew: {rew_comp['distance_improvement']:.3f}\n\
        Jackknife_Pen: {rew_comp['jackknife_penalty']:.3f}"
        self.world.debug.draw_string(
            hud_location,
            text,
            draw_shadow=True,
            color=carla.Color(r=0, g=50, b=0) if self.current_stage == 0 else carla.Color(r=255, g=165, b=0),
            life_time=0.05,
        )
        # endregion
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_jackknife_penalty(self, jackknife_angle):
        if jackknife_angle > 0.73 or jackknife_angle < 0.26:
            return -0.65 * np.cos(-7.1 * jackknife_angle + 3.55)
        else:
            return 0

    def _calculate_reward(self, obs):

        distance = float(obs["distance_to_target"][0])
        angle_diff = abs(float(obs["angle_difference"][0]))
        jackknife = float(obs["jackknife_angle"][0])
        
        # Proximity reward
        proximity_reward = 0.5 * (1 - (distance /  self.init_distance) ** 2)
        
        # Reward for alignment
        y_pre = (1 - distance) * np.cos(2 * math.pi * angle_diff - math.pi)
        alignment_reward = max(0, 0.3 * y_pre) + min(0, 0.15 * y_pre)
        
        # === 3. Jackknife Penalty ===
        jackknife_penalty = self._calculate_jackknife_penalty(jackknife)

        # === 4. Stage Completion Bonus ===
        stage_bonus = 0.0
        
        # === 5. Small time penalty to encourage efficiency ===
        time_penalty = 0.06

        # === Combined Reward ===
        total_reward = (
            proximity_reward +
            alignment_reward +
            stage_bonus -
            jackknife_penalty -
            time_penalty
        )

        info = {}
        info["reward_comp"] = {
            "total_reward": total_reward,
            "proximity_reward": proximity_reward,
            "alignment_reward": alignment_reward,
            "stage_bonus": stage_bonus,
            "jackknife_penalty": jackknife_penalty,
            "time_penalty": time_penalty
        }
        
        return total_reward, info

    def destroy_actors(self):
        """Helper function to destroy all spawned actors."""
        print(f"Destroying {len(self.actor_list)} actors...")
        if not self.actor_list:
            return

        alive_actors = []
        sensor_actors = []

        # Build a stable list first so we can teardown sensors before destruction.
        for actor in self.actor_list:
            if actor is None:
                continue
            try:
                if not actor.is_alive:
                    continue
                alive_actors.append(actor)
                if "sensor" in actor.type_id:
                    sensor_actors.append(actor)
            except RuntimeError:
                # Actor may already be gone on server side.
                continue

        # Stop sensor streams before sending destroy commands.
        for sensor in sensor_actors:
            try:
                sensor.stop()
            except RuntimeError:
                pass

        # Let the server process stop requests before destruction.
        if sensor_actors:
            try:
                self._safe_world_tick(context="destroy_actors", retries=1)
            except RuntimeError:
                pass

        # Batch destroy is less prone to hanging than per-actor RPC destroy calls.
        if alive_actors:
            try:
                destroy_cmds = [carla.command.DestroyActor(actor.id) for actor in alive_actors]
                responses = self.client.apply_batch_sync(destroy_cmds, True)

                for actor, response in zip(alive_actors, responses):
                    if response.error:
                        print(f"DestroyActor failed for id={actor.id} type={actor.type_id}: {response.error}")
            except RuntimeError as e:
                print(f"Batch sync destroy failed, trying non-blocking batch destroy: {e}")
                try:
                    
                    self.client.apply_batch([carla.command.DestroyActor(actor.id) for actor in alive_actors])
                except RuntimeError as async_error:
                    print(f"Non-blocking batch destroy also failed: {async_error}")
                    kill_carla()

        self.actor_list.clear()
    
    def close(self):
        """
        Closes the environment and destroys all actors.
        """
        self.destroy_actors()
        # --- FIX 2: Add a small delay to prevent sensor warnings ---
        time.sleep(2.0)
    
    def obstacle_callback(self, obstacle_event, queue_id):

        try:
            self.obstacle_queues[queue_id].put(obstacle_event.distance)
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
            print(f"Error in obstacle callback: {e}")

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
