import random
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import A2C
import os
import pandas as pd


# finrl
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import data_split

from preprocess_data import download_stock_data, analyze_trends, format_data
from config import *


def tuple_to_tensor(input_tuple):
    # for each element in the tuple, fetch element[0]
    output = []
    for i in range(len(input_tuple)):
        d = []
        for j in range(len(input_tuple[i])):
            d.append(input_tuple[i][j][0].squeeze())
        output.append(torch.stack(d))

    return output


def collect_trajectories(env, model, num_steps=1000):
    """
    Collect trajectories for a certain number of steps or episodes.

    Parameters:
    - envs: The environment to interact with
    - policy: The policy function to determine actions
    - value_function: The value function to evaluate states
    - num_steps: The number of steps to collect data for

    Returns:
    - logits_list: The log probabilities of actions taken
    - values_list: The value function predictions
    - rewards_list: The observed rewards
    - actions_list: The actions taken
    """
    logits_list = []
    values_list = []
    rewards_list = []
    actions_list = []

    state = env.reset()
    state = np.array(state[0]).reshape((1, -1))
    for step in range(num_steps):

        # handle the case where the state is a tuple
        if isinstance(state, tuple):
            state = np.array(state[0]).reshape((1, -1))


        state_tensor = torch.from_numpy(np.array(state)).float()
        action = model.policy.predict(state)

        state_tensor = state_tensor.unsqueeze(0)
        value = model.policy.predict_values(state_tensor.to(model.device))

        logits = model.policy.forward(state_tensor.to(model.device))

        action_input = np.array(action[0]).flatten()
        next_state, reward, done, _, _ = env.step(action_input)
        reward = torch.tensor(reward).float()

        # Store the outputs for loss calculation
        logits_list.append(logits)
        values_list.append(value)
        rewards_list.append(reward)
        actions_list.append(torch.tensor(action[0]).squeeze())# Convert to tensor

        state = next_state if not done else env.reset()

    # Convert lists to tensors
    logits_list = tuple_to_tensor(logits_list)

    logits_tensor = torch.stack(logits_list)
    values_tensor = torch.stack(values_list)
    rewards_tensor = torch.stack(rewards_list)
    actions_tensor = torch.stack(actions_list)

    return logits_tensor, values_tensor, actions_tensor, rewards_tensor


def sample_tasks(upward_segments, downward_segments, bumpy_segments, num_tasks, env_kwargs):
    # Randomly sample segments from each trend category
    sampled_segments = []
    sampled_segments += random.choices(upward_segments, k=num_tasks // 3)
    sampled_segments += random.choices(downward_segments, k=num_tasks // 3)
    sampled_segments += random.choices(bumpy_segments, k=num_tasks - 2 * (num_tasks // 3))

    # Create a new environment instance for each segment
    envs = [StockTradingEnv(segment.reset_index(drop=True), **env_kwargs) for segment in sampled_segments]
    return envs, sampled_segments


def compute_loss(logits, values, actions, rewards, gamma=0.99):
    """
    Compute the A2C loss given the model's logits, values, taken actions, and observed rewards.

    Parameters:
    - logits: The log probabilities of the actions from the policy network
    - values: The predicted values from the value network
    - actions: The actions taken
    - rewards: The observed rewards after taking those actions
    - gamma: The discount factor for future rewards

    Returns:
    - total_loss: The total loss for the A2C model
    """
    # Calculate the advantages
    advantages = rewards - values
    advantages = advantages.squeeze()

    # Calculate the policy loss
    action_log_probs = F.log_softmax(logits, dim=1)
    action_probs = torch.exp(action_log_probs)

    # find action_taken_log_probs from the max value of each row in action_log_probs
    action_taken_log_probs = action_probs.max(dim=1).values
    policy_loss = -torch.mean(action_taken_log_probs * advantages)

    # Calculate the value loss
    returns = rewards
    for t in reversed(range(len(rewards) - 1)):
        returns[t] = returns[t] + gamma * returns[t + 1]
    value_loss = F.mse_loss(values.squeeze(), returns)

    # Combine the two parts of the loss
    total_loss = policy_loss + value_loss

    return total_loss.float()


def maml_model(df, stock_idx, epochs, num_tasks, beta):
    # Split dataset
    tickers = df["tic"].unique().tolist()
    target_tic = tickers[stock_idx]
    df_train = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
    df_train = format_data(df_train.loc[df_train["tic"].isin([target_tic])])

    df_trade = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
    df_trade = format_data(df_trade.loc[df_trade["tic"].isin([target_tic])])
    tmp = []
    types = analyze_trends(df_trade, SLOPE_THRESHOLD_UP, SLOPE_THRESHOLD_DOWN)
    for type in types:
        tmp += type
    df_trade = tmp

    upward_segments, downward_segments, bumpy_segments = analyze_trends(df_train, SLOPE_THRESHOLD_UP, SLOPE_THRESHOLD_DOWN)


    # Config
    stock_dimension = 1
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    buy_cost_list = sell_cost_list = [0] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "print_verbosity": 1000000
    }

    # MAML
    dummy_env = StockTradingEnv(upward_segments[0], **env_kwargs)
    model = A2C('MlpPolicy', dummy_env, verbose=1)
    model_param_shape = [list(p.size()) for p in model.policy.parameters()]
    best_return = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        meta_grad = [torch.zeros(param_shape) for param_shape in model_param_shape]

        # Task sampling
        tasks, _ = sample_tasks(upward_segments, downward_segments, bumpy_segments, num_tasks, env_kwargs)

        # Inner loop
        for task_env in tasks:

            # Initialize parameters
            model_copy = A2C('MlpPolicy', task_env, verbose=1)
            theta = list(model.policy.parameters())
            with torch.no_grad():
                i = 0
                for p in model_copy.policy.parameters():
                    p.copy_(theta[i])
                    i += 1

            model_copy.learn(total_timesteps=300)

            # evaluate the model with the new parameters
            logits, values, actions, returns = collect_trajectories(task_env, model_copy)
            loss = compute_loss(logits, values, actions, returns)

            # manually sum all gradient
            grads = torch.autograd.grad(loss, list(model_copy.policy.parameters()))
            for i in range(len(meta_grad)):
                meta_grad[i] += grads[i]

            # delete the model_copy
            del model_copy

        # Outer loop
        # Compute the meta gradient and update theta
        theta = list(model.policy.parameters())
        with torch.no_grad():
            for i in range(len(theta)):
                theta[i] -= meta_grad[i] * beta

            i = 0
            for p in model.policy.parameters():
                p.copy_(theta[i])
                i += 1

        # Manually clear model's gradient
        model.policy.zero_grad()

        # Testing
        if epoch % 5 == 0:
            final_asset = []
            for segment in df_trade:
                # Meta-testing
                env = StockTradingEnv(df=segment, **env_kwargs)
                # Initialize parameters
                model_copy = A2C('MlpPolicy', env, verbose=1)
                theta = list(model.policy.parameters())
                with torch.no_grad():
                    i = 0
                    for p in model_copy.policy.parameters():
                        p.copy_(theta[i])

                model_copy.learn(total_timesteps=300)

                test_env, test_obs = env.get_sb_env()
                test_env.reset()
                with torch.no_grad():
                    for i in range(len(segment.index.unique())):
                        action, state, = model.predict(test_obs)
                        test_obs, rewards, done, info = test_env.step(action)

                        if i == len(segment.index.unique()) - 2:
                            account_memory = test_env.env_method(method_name="save_asset_memory")
                            final_asset.append(account_memory[0]["account_value"].tolist()[-1])
            if sum(final_asset) / len(final_asset) > best_return:
                print(f"Saving best model: {sum(final_asset) / len(final_asset)}")
                model.save(f"trained_models/A2C/MAML_{target_tic}")




if __name__ == "__main__":
    if not os.path.exists("stock.csv"):
        df = download_stock_data(TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE)
        df.to_csv("stock.csv")
    else:
        df = pd.read_csv("stock.csv")

    for i in range(8):
        maml_model(df, i, 300, 20, 0.01)
