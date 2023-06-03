import os
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback,StopTrainingOnNoModelImprovement
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv




if __name__ == "__main__":

    models_dir = "models/PP0_CNN_ORIGINAL"
    logdir = "logs"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)



    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)


    class CustomCNN(BaseFeaturesExtractor):
        """
        Custom CNN architecture for feature extraction.
        """

        def __init__(self, observation_space: gym.spaces.Discrete, features_dim: int = 128):
            super().__init__(observation_space, features_dim)
            # We assume CxHxW images (channels first)
            # Re-ordering will be done by pre-preprocessing or wrapper
            n_input_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 128, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=4, stride=1, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

            # Compute shape by doing one forward pass
            with torch.no_grad():
                n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

            self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.linear(self.cnn(observations))


    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128)
    )

    # Create the base environment
    base_env = make_atari_env("ALE/AirRaid-v5",
                            n_envs=8,
                            seed=21,
                            # vec_env_cls=SubprocVecEnv
    )

    # Frame-stacking with 4 frames
    train_env = VecFrameStack(base_env, n_stack=4)

    # Separate evaluation env with the same base environment
    eval_env = VecFrameStack(base_env, n_stack=4)


    eval_callback = EvalCallback(eval_env, 
                                best_model_save_path="./logs/best_model", 
                                log_path="./logs/results", 
                                eval_freq=1000, 
                                verbose=1
    )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/")
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Create the PPO agent with the custom network
    model = PPO(
        'CnnPolicy',
        train_env,
        # policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=logdir,
        device=device,
        learning_rate=10e-5,
        seed=21
    )

    TIMESTEPS = 200000
    
    model.learn(total_timesteps=TIMESTEPS,
                progress_bar=True, 
                callback=callback
        )
    model.save(f"{models_dir}/{TIMESTEPS}/")

    print("Training done")

    mean_reward, std_reward = evaluate_policy(model, eval_env , n_eval_episodes=10)
    print("Mean Reward:", mean_reward)
    print("Std Reward:", std_reward)

    train_env.metadata['render_fps'] = 29

    # Create gif
    images = []
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")

    for i in range(300):
        images.append(img)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = model.env.step(action)
        img = model.env.render(mode="rgb_array")

    # Save the GIF
    imageio.mimsave("air-raid_cnn_original.gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], duration=500)
