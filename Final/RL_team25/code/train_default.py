import pandas as pd
import os
from tqdm import tqdm
import random
import torch
from stable_baselines3 import PPO, A2C

# FinRL
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import data_split
from preprocess_data import download_stock_data, analyze_trends, format_data
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from config import *


def train(df: pd.DataFrame, model_path: str):
    # Split dataset
    tickers = df["tic"].unique().tolist()
    df_train = data_split(df, TRAIN_START_DATE, TRAIN_END_DATE)
    df_train = format_data(df_train)
    df_trade = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
    df_trade = format_data(df_trade)

    tmp = []
    for i in range(len(tickers)):
        trends = analyze_trends(df_train[df_train["tic"].isin([tickers[i]])], SLOPE_THRESHOLD_UP, SLOPE_THRESHOLD_DOWN)
        for trend in trends:
            tmp += trend
    df_train = tmp
    df_train_len = len(df_train)

    # Config
    stock_dimension = 1
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    buy_cost_list = sell_cost_list = [0] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs = {
        "hmax": 10000,
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
    print(f"Stock dimension: {stock_dimension}, State space: {state_space}, #segments: {df_train_len}")

    A2C_PARAMS = {
        "n_steps": 90,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
    }

    # train
    best_return = 0
    num_epoch = 20
    envs = [StockTradingEnv(df=df_train[i], **env_kwargs) for i in range(df_train_len)]
    model = A2C(policy="MlpPolicy", env=envs[0], **A2C_PARAMS)
    state_dict = model.policy.state_dict()
    for epoch in range(num_epoch):
        # Random training sequence
        indexs = random.sample(list(range(df_train_len)), int(df_train_len / 2))
        segments = tqdm([envs[i] for i in indexs])
        segments.set_description(f"Epoch {epoch}")

        # Learn
        for env in segments:
            model = A2C(policy="MlpPolicy", env=env, **A2C_PARAMS)
            model.policy.load_state_dict(state_dict)

            model.learn(total_timesteps=270)
            state_dict = model.policy.state_dict()

        # Testing
        avg_return = 0
        for i in range(len(tickers)):
            cur_df_trade = df_trade[df_trade["tic"].isin([tickers[i]])]
            env = StockTradingEnv(df=cur_df_trade, **env_kwargs)
            test_env, test_obs = env.get_sb_env()
            test_env.reset()
            with torch.no_grad():
                for j in range(len(cur_df_trade.index.unique())):
                    action, state, = model.predict(test_obs)
                    test_obs, rewards, done, info = test_env.step(action)

                    if j == len(df_trade.index.unique()) - 2:
                        account_memory = test_env.env_method(method_name="save_asset_memory")
                        avg_return += account_memory[0]["account_value"].tolist()[-1]
        avg_return /= len(tickers)
        if avg_return > best_return:
            model.save(model_path)
            best_return = avg_return

        print(f"Epoch: {epoch}, Final asset: {avg_return}")


if __name__ == "__main__":
    dirs = [TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    if not os.path.exists("stock.csv"):
        df = download_stock_data(TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE)
        df.to_csv("stock.csv")
    else:
        df = pd.read_csv("stock.csv")

    train(df, os.path.join("trained_models/A2C", "baseline"))
