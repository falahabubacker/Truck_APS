import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CustomMetricsCallback(BaseCallback):
    """
    Custom callback to log parking-specific metrics to TensorBoard
    """

    def __init__(self, verbose=0, noise_sigma_init=None, noise_sigma_final=None, noise_decay_steps=None):
        super().__init__(verbose)
        self.episode_distances = []
        self.episode_angles = []
        self.episode_jackknifes = []
        self.episode_phis = []
        self.episode_stages = []
        # region individual reward components
        self.episode_reward_components = {
            "total_reward": 0.0,
            "angle_improvement": 0.0,
            "distance_improvement": 0.0,
            "proximity_reward": 0.0,
            "alignment_reward": 0.0,
            "stage_bonus": 0.0,
            "jackknife_penalty": 0.0,
            "time_penalty": 0.0}
        # endregion
        self.episode_cumulative_reward = 0.0
        self.stage_1_completions = 0
        self.stage_2_successes = 0
        # region newly added
        self.noise_sigma_init = noise_sigma_init
        self.noise_sigma_final = noise_sigma_final
        self.noise_decay_steps = noise_decay_steps
        # endregion
        
    def _on_step(self) -> bool:
        # Get current episode number from environment at every step
        try:
            env = self.model.env.envs[0].unwrapped
            current_episode = env.episode_number
            # Update logger's step counter to use episode number for ALL metrics
            # This ensures the x-axis in TensorBoard represents actual episodes
            self.logger.num_timesteps = current_episode
        except (AttributeError, IndexError):
            current_episode = 0
        
        # Accumulate reward for the current step
        rewards = self.locals.get('rewards', [0.0])

        # Get the current observation from the environment
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
        
        if len(rewards) > 0:
            self.episode_cumulative_reward += rewards[0]
            self.episode_reward_components["total_reward"] += rewards[0]
            self.episode_reward_components["angle_improvement"] += info["reward_comp"].get("angle_improvement")
            self.episode_reward_components["distance_improvement"] += info["reward_comp"].get("distance_improvement")
            self.episode_reward_components["proximity_reward"] += info["reward_comp"].get("proximity_reward")
            self.episode_reward_components["alignment_reward"] += info["reward_comp"].get("alignment_reward")
            self.episode_reward_components["stage_bonus"] += info["reward_comp"].get("stage_bonus")
            self.episode_reward_components["jackknife_penalty"] += info["reward_comp"].get("jackknife_penalty")
            self.episode_reward_components["time_penalty"] += info["reward_comp"].get("time_penalty")
            
            # Check if episode is done
            if self.locals.get('dones', [False])[0]:
                # Log episode number as a reference metric
                self.logger.record('episode_number', current_episode)
                
                # Log episode statistics with episode number as the x-axis
                self.logger.record('rollout/ep_cum_rew', self.episode_cumulative_reward)
                
                if len(self.episode_distances) > 0:
                    self.logger.record('metrics/mean_distance', np.mean(self.episode_distances))
                    self.logger.record('metrics/final_distance', self.episode_distances[-1])
                    self.logger.record('metrics/min_distance', np.min(self.episode_distances))
                    
                if len(self.episode_angles) > 0:
                    self.logger.record('metrics/mean_angle_error', np.mean(self.episode_angles))
                    self.logger.record('metrics/final_angle_error', self.episode_angles[-1])
                    
                if len(self.episode_jackknifes) > 0:
                    self.logger.record('metrics/mean_jackknife', np.mean(self.episode_jackknifes))
                    self.logger.record('metrics/max_jackknife', np.max(np.abs(self.episode_jackknifes)))
                    
                if len(self.episode_phis) > 0:
                    self.logger.record('metrics/mean_phi', np.mean(self.episode_phis))
                    
                if len(self.episode_stages) > 0:
                    stage_1_reached = any(s == 1 for s in self.episode_stages)
                    if stage_1_reached:
                        self.stage_1_completions += 1
                    self.logger.record('progress/stage_1_completion_rate', 
                                     self.stage_1_completions / max(1, current_episode))
                
                # Dump metrics with episode number as the step counter for TensorBoard
                # This makes the "Step" axis in TensorBoard represent episodes
                self.logger.dump(current_episode)
                
                # Reset episode tracking
                self.episode_distances = []
                self.episode_angles = []
                self.episode_jackknifes = []
                self.episode_phis = []
                self.episode_stages = []
                self.episode_cumulative_reward = 0.0
        
        # Track metrics during episode (from flattened observation)
        obs = self.locals.get('new_obs', None)
        if obs is not None and len(obs) > 0:
            # Extract metrics from flattened observation
            # Indices depend on observation structure: [truck_pose(3), trailer_pose(3), parking_pose(3), 
            # truck_parking_pose(3), current_stage(1), distance(1), angle_diff(1), jackknife(1), phi(1), 
            # parallel_dist(1), longitudinal_dist(1), radar_data(30)]
            # Total: 3+3+3+3+1+1+1+1+1+1+1+30 = 48
            
            distance = obs[0][13] if len(obs[0]) > 13 else None  # distance_to_target
            angle_diff = obs[0][14] if len(obs[0]) > 14 else None  # angle_difference
            jackknife = obs[0][15] if len(obs[0]) > 15 else None  # jackknife_angle
            phi = obs[0][16] if len(obs[0]) > 16 else None  # phi
            stage = obs[0][12] if len(obs[0]) > 12 else None  # current_stage
            
            if distance is not None:
                self.episode_distances.append(distance)
            if angle_diff is not None:
                self.episode_angles.append(abs(angle_diff))
            if jackknife is not None:
                self.episode_jackknifes.append(jackknife)
            if phi is not None:
                self.episode_phis.append(phi)
            if stage is not None:
                self.episode_stages.append(stage)
        
        return True
