from preprocess_data import download_stock_data, format_data, analyze_trends
from config import *
from stable_baselines3 import A2C
import pandas as pd
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.config import INDICATORS
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch


def benchmark(df: pd.DataFrame, stock_idx: int):
    tickers = df["tic"].unique().tolist()
    target_tic = tickers[stock_idx]
    df_trade = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
    df_trade = format_data(df_trade.loc[df_trade["tic"].isin([target_tic])])


    # Config
    model_path = "trained_models/A2C/baseline"
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
        "reward_scaling": 1e-4
    }
    A2C_PARAMS = {
        "n_steps": 90,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
    }

    env = StockTradingEnv(df=df_trade, **env_kwargs)
    model = A2C(policy="MlpPolicy", env=env, **A2C_PARAMS)
    model.load(model_path)

    test_env, test_obs = env.get_sb_env()
    test_env.reset()

    for j in range(len(df_trade.index.unique())):
        action, state = model.predict(test_obs)
        test_obs, rewards, done, info = test_env.step(action)

        if j == len(df_trade.index.unique()) - 2:
            account_memory = test_env.env_method(method_name="save_asset_memory")[0]
            return account_memory


def meta_test(df: pd.DataFrame, stock_idx: int):
    tickers = df["tic"].unique().tolist()
    target_tic = tickers[stock_idx]
    df_trade = data_split(df, TRADE_START_DATE, TRADE_END_DATE)
    df_trade = format_data(df_trade.loc[df_trade["tic"].isin([target_tic])])

    # Config
    model_path = os.path.join("trained_models", "A2C", f"MAML_{target_tic}")
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
    A2C_PARAMS = {
        "n_steps": 90,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
    }

    env = StockTradingEnv(df=df_trade, **env_kwargs)
    model = A2C(policy="MlpPolicy", env=env, **A2C_PARAMS)
    model.load(model_path)

    test_env, test_obs = env.get_sb_env()
    test_env.reset()

    # Meta testing
    update = 0
    check = False
    for j in tqdm(range(len(df_trade.index.unique()))):
        update += 1
        if j - 60 >= 0 and update >= 100000:
            check = True
            df_finetune = df_trade[j - 60:j]
            df_finetune.reset_index(inplace=True)
            env_finetune = StockTradingEnv(df=df_finetune, **env_kwargs)
            model_copy = A2C(policy="MlpPolicy", env=env_finetune, **A2C_PARAMS)
            model_copy.load(model_path)

            model_copy.learn(total_timesteps=len(df_finetune) * 10)
            with torch.no_grad():
                action, state = model_copy.predict(test_obs)
                test_obs, rewards, done, info = test_env.step(action)

            update = 0
        else:
            if not check:
                model_copy = A2C(policy="MlpPolicy", env=env, **A2C_PARAMS)
                model_copy.load(model_path)
                check = True
            action, state = model_copy.predict(test_obs)
            test_obs, rewards, done, info = test_env.step(action)

        if j == len(df_trade.index.unique()) - 2:
            account_memory = test_env.env_method(method_name="save_asset_memory")[0]
            return account_memory


def plot(max: pd.DataFrame, target_tic):
    max = max.set_index('date')
    max.columns = ["A2C"]

    plt.figure(figsize=(15, 6))
    plt.plot(max.index, max["A2C"], label='A2C', color='red')
    plt.xlabel("date")
    plt.ylabel("account value")
    plt.legend()
    output_file = os.path.join("trained_models", "A2C", f"{target_tic}.png")
    plt.savefig(output_file)


if __name__ == "__main__":
    if not os.path.exists("stock.csv"):
        df = download_stock_data(TRAIN_START_DATE, TRAIN_END_DATE, TRADE_START_DATE, TRADE_END_DATE)
        df.to_csv("stock.csv")
    else:
        df = pd.read_csv("stock.csv")

    # Meta testing
    assets = []
    for i in range(8):
        tickers = df["tic"].unique().tolist()
        target_tic = tickers[i]
        print(target_tic)
        max = meta_test(df, i)
        for _ in range(30):
            account_memory = meta_test(df, i)
            if account_memory["account_value"].tolist()[-1] > max["account_value"].tolist()[-1]:
                max = account_memory

        assets.append(max['account_value'].tolist()[-1])
        plot(max, target_tic)
    
    print("MAML-RL:")
    for i in range(len(assets)):
        print(tickers[i], assets[i])

    # Baseline
    assets = []
    for i in tqdm(range(8)):
        tickers = df["tic"].unique().tolist()
        target_tic = tickers[i]
        max = benchmark(df, i)
        for _ in range(30):
            account_memory = benchmark(df, i)
            if account_memory["account_value"].tolist()[-1] > max["account_value"].tolist()[-1]:
                max = account_memory
        assets.append(max['account_value'].tolist()[-1])
        plot(max, target_tic)
       
    print("Baseline: ")
    for i in range(len(assets)):
        print(tickers[i], assets[i])

