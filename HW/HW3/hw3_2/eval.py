import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, DQN,A2C

import numpy as np
from collections import Counter

register(
    id='2048-eval',
    entry_point='envs:Eval2048Env'
)

def evaluation(env, model, render_last, eval_num=100):
    score = []
    highest = []

    ### Run eval_num times rollouts
    for seed in range(eval_num):
        done = False
        # Set seed and reset env using Gymnasium API
        obs, info = env.reset(seed=seed)

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

        # Render the last board state of each episode
        # print("Last board state:")
        # env.render()

        score.append(info['score'])
        highest.append(info['highest'])

    ### Render last rollout
    if render_last:
        print("Rendering last rollout")
        done = False
        obs, info = env.reset(seed=eval_num-1)
        env.render()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            env.render()

        
    return score, highest


if __name__ == "__main__":
    model_path = "models/A2C_test_fin/367"  # Change path name to load different models
    env = gym.make('2048-eval')

    ### Load model with SB3
    # Note: Model can be loaded with arbitrary algorithm class for evaluation
    # (You don't necessarily need to use PPO for training)
    model = A2C.load(model_path)
    
    eval_num = 100
    score, highest = evaluation(env, model, True, eval_num)

    print("Avg_score:  ", np.sum(score)/eval_num)
    print("Avg_highest:", np.sum(highest)/eval_num)


    print(f"Counts: (Total of {eval_num} rollouts)")
    c = Counter(highest)
    for item in (sorted(c.items(),key = lambda i: i[0])):
        print(f"{item[0]}: {item[1]}")
    
    # draw the histogram
    import matplotlib.pyplot as plt
    # x values for the histogram discrete and only for all c.keys()
    # print(c)
    # plt.bar(list(c.keys()), list(c.values()), color='g',width=20)
    # plt.xticks(list(c.keys())) 
    # plt.xlabel('Highest Tile')
    # plt.ylabel('Count')
    # plt.title('Histogram of Highest Tile')

    # # save the figure
    # plt.savefig('histogram3.png')
