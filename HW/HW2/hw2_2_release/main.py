import random
import numpy as np

from DP_solver_2_2 import (
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

STEP_REWARD       = -0.1
GOAL_REWARD       = 1.0
TRAP_REWARD       = -1.0
DISCOUNT_FACTOR   = 0.99
LEARNING_RATE     = 0.01
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500


def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt"):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world

def run_MC_policy_iteration(grid_world: GridWorld, iter_num: int):
    print(bold(underline("MC Policy Iteration")))
    policy_iteration = MonteCarloPolicyIteration(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"MC Policy Iteration",
        show=False,
        filename=f"MC_policy_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

def run_SARSA(grid_world: GridWorld, iter_num: int):
    print(bold(underline("SARSA Policy Iteration")))
    policy_iteration = SARSA(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"SARSA",
        show=False,
        filename=f"SARSA_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

def run_Q_Learning(grid_world: GridWorld, iter_num: int):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"Q_Learning",
        show=False,
        filename=f"Q_Learning_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()



if __name__ == "__main__":
    # wandb.init(project="RL-GridWorld")

    grid_world = init_grid_world()
    
    # for eps in epsilons:
    #     print(f'epsilon={eps}')
    #     reward_record, loss_record = run_MC_policy_iteration(grid_world, MC_and_TD_episode,epsilon=eps)
    #     avg_reward = [sum(row)/len(row) for row in reward_record]
    #     avg_r = [np.mean(avg_reward[i:i+10]) for i in range(0, len(avg_reward))]
    #     # avg_loss = [sum(row)/len(row) for row in loss_record]
    #     avg_l = [np.mean(loss_record[i:i+10]) for i in range(0, len(loss_record))]

    #     plt.figure(figsize=(20, 10))
    #     plt.plot(np.arange(len(avg_r)), np.array(avg_r), label='reward')
    #     plt.legend()
    #     plt.xlabel('episode')
    #     plt.ylabel('reward')
    #     plt.title(f'MC_epsilon_reward={eps}')
    #     plt.savefig(f'MC_epsilon_reward={eps}.png')
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(np.arange(len(avg_l)), np.array(avg_l), label='loss')
    #     plt.xlabel('episode')
    #     plt.ylabel('loss')
    #     plt.title(f'MC_epsilon_loss={eps}')
    #     plt.savefig(f'MC_epsilon_loss={eps}.png')

    #     reward_record,loss_record = run_SARSA(grid_world, MC_and_TD_episode,epsilon=eps)
    #     avg_reward = [sum(row)/len(row) for row in reward_record]
    #     avg_r = [np.mean(avg_reward[i:i+10]) for i in range(0, len(avg_reward))]
    #     avg_loss = [sum(row)/len(row) for row in loss_record]
    #     avg_l = [np.mean(avg_loss[i:i+10]) for i in range(0, len(avg_loss))]
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(np.arange(len(avg_r)), np.array(avg_r), label='reward')
    #     plt.legend()
    #     plt.xlabel('episode')
    #     plt.ylabel('reward')
    #     plt.title(f'TD_epsilon_reward={eps}')
    #     plt.savefig(f'TD_epsilon_reward={eps}.png')
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(np.arange(len(avg_l)), np.array(avg_l), label='loss')
    #     plt.xlabel('episode')
    #     plt.ylabel('loss')
    #     plt.title(f'TD_epsilon_loss={eps}')
    #     plt.savefig(f'TD_epsilon_loss={eps}.png')
    #     reward_record,loss_record = run_Q_Learning(grid_world, Q_episode,epsilon=eps)
    #     avg_reward = [sum(row)/len(row) for row in reward_record]
    #     avg_r = [np.mean(avg_reward[i:i+10]) for i in range(0, len(avg_reward))]
    #     avg_loss = [sum(row)/len(row) for row in loss_record]
    #     avg_l = [np.mean(avg_loss[i:i+10]) for i in range(0, len(avg_loss))]
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(np.arange(len(avg_r)), np.array(avg_r), label='reward')
    #     plt.legend()
    #     plt.xlabel('episode')
    #     plt.ylabel('reward')
    #     plt.title(f'Q_epsilon_reward={eps}')
    #     plt.savefig(f'Q_epsilon_reward={eps}.png')
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(np.arange(len(avg_l)), np.array(avg_l), label='loss')
    #     plt.xlabel('episode')
    #     plt.ylabel('loss')
    #     plt.title(f'Q_epsilon_loss={eps}')
    #     plt.savefig(f'Q_epsilon_loss={eps}.png')
    # sample_batch_sizes = [100,500,1000]
    
    # for sample_batch_size in sample_batch_sizes:
    #     reward_record,loss_record = run_Q_Learning(grid_world, Q_episode,epsilon=0.2,sample_batch_size=sample_batch_size)
    #     avg_reward = [sum(row)/len(row) for row in reward_record]
    #     avg_r = [np.mean(avg_reward[i:i+10]) for i in range(0, len(avg_reward))]
    #     avg_loss = [sum(row)/len(row) for row in loss_record]
    #     avg_l = [np.mean(avg_loss[i:i+10]) for i in range(0, len(avg_loss))]
    #     plt.figure(figsize=(20, 10))
    #     plt.plot(np.arange(len(avg_l)), np.array(avg_l), label=f'sample_batch_size={sample_batch_size}')
    #     plt.legend()
    #     plt.xlabel('episode')
    #     plt.ylabel('loss')
    #     plt.title(f'Q_epsilon_loss, sample_batch_size={sample_batch_size}')
    #     plt.savefig(f'Q_epsilon_loss,sample_batch_size={sample_batch_size}.png')
    # run_MC_policy_iteration(grid_world, 512000)
    run_SARSA(grid_world, 512000)
    # run_Q_Learning(grid_world, 50000)

