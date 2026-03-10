from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FlattenObservation
from parking_env_3 import ParkingLotEnv
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from metrics_callback import CustomMetricsCallback
import os
import json
import torch
import torch.nn as nn

# # --- Configuration ---
LOG_DIR = "logs/"
MODEL_SAVE_PATH = "models/td3_parking/"
EPISODE_STATE_FILE = os.path.join(MODEL_SAVE_PATH, "episode_state.json")
TOTAL_TIMESTEPS = 1000000  # Increased for complex parking task
LEARNING_STARTS = 20000  # More random exploration to fill buffer
BUFFER_SIZE = 100000  # Replay buffer size from Table 6
BATCH_SIZE = 128  # Larger batch for more stable critic with scaled rewards
SAVE_FREQ = 25000  # Save less frequently with longer training

# Hyperparameters from Table 6 (TD3)
ACTOR_LR = 0.0001  # Actor network learning rate
CRITIC_LR = 0.001  # Critic network learning rate
GAMMA = 0.95  # Increased discount for long-horizon task
TAU = 0.001  # Slower target updates for stability
NOISE_SIGMA = 0.4  # Reduced exploration noise for simpler state
NOISE_SIGMA_FINAL = 0.03  # Final exploration noise after decay
NOISE_DECAY_STEPS = 300000  # Timesteps to linearly decay noise
NOISE_THETA = 0.15  # Ornstein-Uhlenbeck noise theta
NOISE_DT = 0.01  # Ornstein-Uhlenbeck noise dt

# TD3-specific parameters
GRADIENT_STEPS = 2  # Multiple gradient steps per env step for efficiency

# # Create directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def save_episode_state(env, episode_state_file):
    """Save the episode state to a JSON file."""
    try:
        env_unwrapped = env.envs[0].unwrapped
        state = env_unwrapped.get_episode_state()
        with open(episode_state_file, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"Warning: Could not save episode state: {e}")

def load_episode_state(episode_state_file):
    """Load the episode state from a JSON file."""
    if os.path.exists(episode_state_file):
        try:
            with open(episode_state_file, 'r') as f:
                state = json.load(f)
            print(f"Episode state loaded: {state}")
            return state
        except Exception as e:
            print(f"Warning: Could not load episode state: {e}")
    return None

class EpisodeStateCheckpointCallback(CheckpointCallback):
    """Custom checkpoint callback that also saves episode state."""
    def __init__(self, episode_state_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_state_file = episode_state_file
    
    def _on_step(self) -> bool:
        """Save model and episode state at checkpoint intervals."""
        should_save = super()._on_step()
        if should_save and self.num_timesteps > 0:
            # Save episode state whenever a checkpoint is saved
            save_episode_state(self.model.env, self.episode_state_file)
        return should_save

def main():
    """
    Main function to initialize the environment, set up the model,
    and start the training process.
    """                                                                                                                                                                                                                                             
    
    def make_env():
        env = ParkingLotEnv()
        env = Monitor(env) # Wrap in Monitor first
        env = FlattenObservation(env) # Then flatten the Dict observation
        return env

    env = DummyVecEnv([make_env])
    
    # Load episode state if resuming from checkpoint (before normalization wrapper)
    episode_state = load_episode_state(EPISODE_STATE_FILE)
    if episode_state:
        env.envs[0].unwrapped.restore_episode_state(episode_state)
    
    # Add observation normalization (critical for TD3 with mixed-scale features)
    # Radar data (0-100), distances (0-50), angles (0-180) need normalization
    env = VecNormalize(
        env,
        norm_obs=True,         # Normalize observations to ~N(0,1)
        norm_reward=True,      # Normalize rewards (scaled 5x in env)
        clip_obs=10,         # Clip normalized obs to [-10, 10]
        clip_reward=10,      # Clip normalized rewards to [-10, 10]
        gamma=GAMMA
    )
    
    print("--- Environment Created and Wrapped ---")
    print(f"Observation Space (Flattened): {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # 2. Set up Callbacks
    # Save a checkpoint of the model every 'SAVE_FREQ' steps
    # Also saves episode state for resuming training
    checkpoint_callback = EpisodeStateCheckpointCallback(
        episode_state_file=EPISODE_STATE_FILE,
        save_freq=SAVE_FREQ,
        save_path=MODEL_SAVE_PATH,
        name_prefix="td3_model"
    )
    
    # Custom metrics callback for TensorBoard
    # region old: CustomMetricsCallback(verbose=1)
    metrics_callback = CustomMetricsCallback(
        verbose=1,
        noise_sigma_init=NOISE_SIGMA,
        noise_sigma_final=NOISE_SIGMA_FINAL,
        noise_decay_steps=NOISE_DECAY_STEPS,
    )
    # endregion
    
    # Combine callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList([checkpoint_callback, metrics_callback])
    
    n_actions = env.action_space.shape[-1]
    # Ornstein-Uhlenbeck noise parameters from Table 6 (TD3)
    # μ=0, σ=0.3, θ=0.15, dt=0.01
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),  # μ = 0
        sigma=NOISE_SIGMA * np.ones(n_actions),  # σ = 0.3
        theta=NOISE_THETA,  # θ = 0.15
        dt=NOISE_DT  # dt = 0.01
    )

    # Network architectures from the provided images:
    # Actor (second image): Input(18) -> 256 -> 128 -> 64 -> Output(2, tanh)
    # Critic (first image): Complex architecture with separate state/action processing
    # Note: SB3 uses a simpler concatenated architecture, so we approximate with [256, 256]
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128, 64],      # Actor: 3 hidden layers (from second image)
            qf=[256, 256]           # Critic: 2 hidden layers (from first image - Hidden Layer 1 & 2)
        ),
        optimizer_class=torch.optim.Adam,  # Use Adam optimizer
        optimizer_kwargs=dict(
            eps=1e-8,  # Adam epsilon for numerical stability
            betas=(0.9, 0.999)  # Adam beta parameters
        )
    )

    # 3. Initialize or Load the TD3 Model
    # Check if there's an existing checkpoint to resume from
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, "td3_parking_final.zip")
    
    resume_training = False
    if os.path.exists(checkpoint_path):
        print(f"--- Loading existing model from {checkpoint_path} ---")
        model = TD3.load(
            checkpoint_path,
            env=env,
            tensorboard_log=LOG_DIR,
        )
        # Restore action noise (not serializable)
        model.action_noise = action_noise
        resume_training = True
        print("--- Model loaded, resuming training ---")
    else:
        print("--- No existing model found, creating new model ---")
        model = TD3(
            policy='MlpPolicy',
            env=env,
            policy_kwargs=policy_kwargs,
            action_noise=action_noise,
            learning_rate=ACTOR_LR,  # Learning rate: 0.0003
            gamma=GAMMA,  # Discount factor: 0.9
            tau=TAU,  # Soft update coefficient: 0.001
            batch_size=BATCH_SIZE,  # Batch size: 32
            buffer_size=BUFFER_SIZE,  # Replay buffer: 100,000
            learning_starts=LEARNING_STARTS,  # Steps to take random actions to fill buffer
            gradient_steps=GRADIENT_STEPS,  # One gradient step per update
            verbose=1,
            tensorboard_log=LOG_DIR  # Directory for TensorBoard logs
        )
        
        # Set separate learning rates for actor and critic
        # Actor LR: 0.0003, Critic LR: 0.003 (10x higher)
        model.actor.optimizer.param_groups[0]['lr'] = ACTOR_LR
        model.critic.optimizer.param_groups[0]['lr'] = CRITIC_LR
        
        print("--- TD3 Model Initialized ---")
        print(f"    Optimizer: Adam")
        print(f"    Loss Function: MSE (Mean Squared Error)")
        print(f"    Actor Learning Rate: {ACTOR_LR}")
        print(f"    Critic Learning Rate: {CRITIC_LR}")
    
    # 4. Start Training
    print("--- Starting Training ---")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,  # Use combined callbacks
            log_interval=10,  # Print stats every 10 episodes
            reset_num_timesteps=not resume_training,  # Continue timestep count when resuming
            tb_log_name="td3_parking"  # Use consistent log name for TensorBoard
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # 5. Clean up and save
        print("--- Training Finished ---")
        model.save(os.path.join(MODEL_SAVE_PATH, "td3_parking_final"))
        # Save episode state for future resuming
        save_episode_state(env, EPISODE_STATE_FILE)
        # Save normalization statistics only when VecNormalize is enabled
        if isinstance(env, VecNormalize):
            env.save(os.path.join(MODEL_SAVE_PATH, "vec_normalize.pkl"))
        print(f"Final model saved to {MODEL_SAVE_PATH}")
        env.close() # This will call destroy_actors()

if __name__ == '__main__':
    main()
