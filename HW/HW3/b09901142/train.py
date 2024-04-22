import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.utils import get_schedule_fn
warnings.filterwarnings("ignore")
register(
    id='2048-v0',
    entry_point='envs:My2048Env'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": A2C,
    "policy_network": "MlpPolicy",
    "save_path": "models/A2C_test_fin",

    "epoch_num": 500,
    "timesteps_per_epoch": 3500,
    "eval_episode_num": 5,
    # "learning_rate": 0.0003,
}

def linear_schedule(initial_value, final_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float) Initial learning rate.
    :param final_value: (float) Final learning rate.
    :return: (function) a function that takes the current progress_remaining (from 1 to 0)
             and computes the learning rate.
    """
    def schedule(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0
        """
        return final_value + (initial_value - final_value) * progress_remaining

    return schedule

# Usage
initial_learning_rate = 0.0003
final_learning_rate = 0.0002

learning_rate_schedule = linear_schedule(initial_learning_rate, final_learning_rate)


def make_env():
    env = gym.make('2048-v0')
    return env

def train(env, model, config):

    current_best = 0
    for epoch in range(config["epoch_num"]):

        ### Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        ### Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score = 0
        avg_highest = 0
        for seed in range(config["eval_episode_num"]):
            done = False

            # Set seed using old Gym API
            env.seed(seed)
            obs = env.reset()

            # Interact with env using old Gym API
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            
            avg_highest += info[0]['highest']/config["eval_episode_num"]
            avg_score   += info[0]['score']/config["eval_episode_num"]
        
        print("Avg_score:  ", avg_score)
        print("Avg_highest:", avg_highest)
        print()
        # wandb.log(
        #     {"avg_highest": avg_highest,
        #      "avg_score": avg_score}
        # )
        

        ### Save best model
        if current_best < avg_score:
            print("Saving Model")
            current_best = avg_score
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")

        print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     id=my_config["run_id"]
    # )

    env = DummyVecEnv([make_env])

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"], 
        env, 
        verbose=1,
        tensorboard_log=my_config["run_id"],
        learning_rate=learning_rate_schedule
        # learning_rate_schedule=learning_rate_schedule

    )
    train(env, model, my_config)