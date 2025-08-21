# 匯入 gymnasium 套件當作 gym，用來建立強化學習環境
import gymnasium as gym
# 匯入 numpy 套件當作 np，用來處理數字陣列
import numpy as np
# 從 gymnasium 匯入 spaces，用來定義觀察和動作空間
from gymnasium import spaces
# 從 stable_baselines3 匯入 PPO，這是強化學習的演算法，用來訓練代理人
from stable_baselines3 import PPO
# 從 stable_baselines3.common.env_checker 匯入 check_env，用來檢查環境有沒有問題
from stable_baselines3.common.env_checker import check_env
# 匯入 random，用來產生隨機數字
import random

# 定義 EVChargingEnv 類別，繼承自 gym.Env，這是電動車充電的環境模擬
class EVChargingEnv(gym.Env):
    # 初始化函數，設定使用者數、時間槽、最大充電功率、電池容量、最大基載參數
    def __init__(self, n_users=5, n_time_slots=24, max_charging_power=50.0, battery_capacity=60.4,
                 max_base_load=220.0):
        # 呼叫父類別的初始化
        super(EVChargingEnv, self).__init__()
        # 設定使用者數量
        self.n_users = n_users
        # 設定時間槽數量，一天24小時
        self.n_time_slots = n_time_slots
        # 最大充電功率，單位是kW
        self.max_charging_power = max_charging_power
        # 電池容量，單位kWh
        self.battery_capacity = battery_capacity
        # 初始電池電量，電池容量的一半
        self.initial_soc = battery_capacity / 2
        # 最大基載功率
        self.max_base_load = max_base_load
        # 目前步驟，從0開始
        self.current_step = 0

        # 使用者資料
        # 充電持續時間清單，一堆數字代表每個使用者的充電小時數
        self.duration = [
            4,6,3,5,7,2,8,6,4,3,
            5,7,2,6,4,8,3,5,7,6,
            4,2,5,3,7
        ]
        # 所需能量清單，每個使用者的充電需求kWh
        self.required_energy = [
            12.5,14.0,11.2,13.8,10.5,12.0,15.0,13.2,11.8,12.6,
            14.4,13.7,10.9,12.3,11.5,14.9,13.1,12.8,10.7,13.4,
            11.9,12.2,13.6,14.1,10.8
        ]
        # 迴圈檢查每個使用者，如果初始電量加需求超過容量，就調整需求減一點點
        for i in range(n_users):
            if self.initial_soc + self.required_energy[i] > self.battery_capacity:
                self.required_energy[i] = self.battery_capacity - self.initial_soc - 0.1

        #  50 組基載，每組 24 時段，這是不同情境的基載功率
        self.n_scenarios = 50
        self.base_load_scenarios = [
            [200,195,190,185,180,185,195,200,190,180,165,158,150,142,137,132,128,122,112,106,108,110,108,106],
            [198,193,188,183,178,183,193,198,188,178,163,156,148,140,135,130,126,120,110,106,108,110,108,106],
            [196,191,186,181,176,181,191,196,186,176,161,154,146,138,133,128,124,118,108,106,108,110,108,106],
            [194,189,184,179,174,179,189,194,184,174,159,152,144,136,131,126,122,116,106,106,108,110,108,106],
            [192,187,182,177,172,177,187,192,182,172,157,150,142,134,129,124,120,114,104,106,108,110,108,106],
            [190,185,180,175,170,175,185,190,180,170,155,148,140,132,127,122,118,112,102,106,108,110,108,106],
            [188,183,178,173,168,173,183,188,178,168,153,146,138,130,125,120,116,110,100,106,108,110,108,106],
            [186,181,176,171,166,171,181,186,176,166,151,144,136,128,123,118,114,108,98,106,108,110,108,106],
            [184,179,174,169,164,169,179,184,174,164,149,142,134,126,121,116,112,106,96,106,108,110,108,106],
            [182,177,172,167,162,167,177,182,172,162,147,140,132,124,119,114,110,104,94,106,108,110,108,106],
            [180,175,170,165,160,165,175,180,170,160,145,138,130,122,117,112,108,102,92,106,108,110,108,106],
            [178,173,168,163,158,163,173,178,168,158,143,136,128,120,115,110,106,100,90,106,108,110,108,106],
            [176,171,166,161,156,161,171,176,166,156,141,134,126,118,113,108,104,98,88,106,108,110,108,106],
            [174,169,164,159,154,159,169,174,164,154,139,132,124,116,111,106,102,96,86,106,108,110,108,106],
            [172,167,162,157,152,157,167,172,162,152,137,130,122,114,109,104,100,94,84,106,108,110,108,106],
            [170,165,160,155,150,155,165,170,160,150,135,128,120,112,107,102,98,92,82,106,108,110,108,106],
            [168,163,158,153,148,153,163,168,158,148,133,126,118,110,105,100,96,90,80,106,108,110,108,106],
            [166,161,156,151,146,151,161,166,156,146,131,124,116,108,103,98,94,88,80,106,108,110,108,106],
            [164,159,154,149,144,149,159,164,154,144,129,122,114,106,101,96,92,86,80,106,108,110,108,106],
            [162,157,152,147,142,147,157,162,152,142,127,120,112,104,99,94,90,84,80,106,108,110,108,106],
            [160,155,150,145,140,145,155,160,150,140,125,118,110,102,97,92,88,82,80,106,108,110,108,106],
            [158,153,148,143,138,143,153,158,148,138,123,116,108,100,95,90,86,80,80,106,108,110,108,106],
            [156,151,146,141,136,141,151,156,146,136,121,114,106,98,93,88,84,80,80,106,108,110,108,106],
            [154,149,144,139,134,139,149,154,144,134,119,112,104,96,91,86,82,80,80,106,108,110,108,106],
            [152,147,142,137,132,137,147,152,142,132,117,110,102,94,89,84,80,80,80,106,108,110,108,106],
            [150,145,140,135,130,135,145,150,140,130,115,108,100,92,87,82,80,80,80,106,108,110,108,106],
            [148,143,138,133,128,133,143,148,138,128,113,106,98,90,85,80,80,80,80,106,108,110,108,106],
            [146,141,136,131,126,131,141,146,136,126,111,104,96,88,83,80,80,80,80,106,108,110,108,106],
            [144,139,134,129,124,129,139,144,134,124,109,102,94,86,81,80,80,80,80,106,108,110,108,106],
            [142,137,132,127,122,127,137,142,132,122,107,100,92,84,79,80,80,80,80,106,108,110,108,106],
            [140,135,130,125,120,125,135,140,130,120,105,98,90,82,77,80,80,80,80,106,108,110,108,106],
            [138,133,128,123,118,123,133,138,128,118,103,96,88,80,75,80,80,80,80,106,108,110,108,106],
            [136,131,126,121,116,121,131,136,126,116,101,94,86,78,73,80,80,80,80,106,108,110,108,106],
            [134,129,124,119,114,119,129,134,124,114,99,92,84,76,71,80,80,80,80,106,108,110,108,106],
            [132,127,122,117,112,117,127,132,122,112,97,90,82,74,69,80,80,80,80,106,108,110,108,106],
            [130,125,120,115,110,115,125,130,120,110,95,88,80,72,67,80,80,80,80,106,108,110,108,106],
            [128,123,118,113,108,113,123,128,118,108,93,86,78,70,65,80,80,80,80,106,108,110,108,106],
            [126,121,116,111,106,111,121,126,116,106,91,84,76,68,63,80,80,80,80,106,108,110,108,106],
            [124,119,114,109,104,109,119,124,114,104,89,82,74,66,61,80,80,80,80,106,108,110,108,106],
            [122,117,112,107,102,107,117,122,112,102,87,80,72,64,59,80,80,80,80,106,108,110,108,106],
            [120,115,110,105,100,105,115,120,110,100,85,78,70,62,57,80,80,80,80,106,108,110,108,106],
            [118,113,108,103,98,103,113,118,108,98,83,76,68,60,55,80,80,80,80,106,108,110,108,106],
            [116,111,106,101,96,101,111,116,106,96,81,74,66,58,53,80,80,80,80,106,108,110,108,106],
            [114,109,104,99,94,99,109,114,104,94,79,72,64,56,51,80,80,80,80,106,108,110,108,106],
            [112,107,102,97,92,97,107,112,102,92,77,70,62,54,49,80,80,80,80,106,108,110,108,106],
            [110,105,100,95,90,95,105,110,100,90,75,68,60,52,47,80,80,80,80,106,108,110,108,106]
        ]
        # 預設使用第一個情境的基礎負載
        self.current_scenario = self.base_load_scenarios[0]

        # State / Action space
        # 計算狀態大小，包括步驟、使用者的電池、開始、充電、剩餘能量、基載
        state_size = 1 + n_users * 4 + n_time_slots
        # 觀察空間，無限大，從0開始，形狀是狀態大小
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(state_size,), dtype=np.float32)
        # 動作空間，從0到1，使用者數*2，因為開始和功率
        self.action_space = spaces.Box(low=0, high=1, shape=(n_users * 2,), dtype=np.float32)

        # 重置環境
        self.reset()

    # 重置環境，可以指定種子和情境索引
    def reset(self, seed=None, scenario_idx=None):
        # 呼叫父類別重置
        super().reset(seed=seed)
        # 步驟歸零
        self.current_step = 0
        # 每個使用者的電池電量重置為初始
        self.battery_levels = [self.initial_soc for _ in range(self.n_users)]
        # 開始充電旗標，重置為False
        self.started = [False for _ in range(self.n_users)]
        # 開始時間，重置為-1
        self.start_times = [-1 for _ in range(self.n_users)]
        # 充電中旗標，重置為False
        self.charging = [False for _ in range(self.n_users)]
        # 剩餘能量，重置為需求複本
        self.remaining_energy = self.required_energy.copy()
        # 如果指定情境，就用那個，否則用第一個
        if scenario_idx is not None:
            self.current_scenario = self.base_load_scenarios[scenario_idx]
        else:
            self.current_scenario = self.base_load_scenarios[0]
        # 回傳觀察和空字典，空字典 {}表示目前沒有額外資訊要提供
        return self._get_obs(), {}

    # 取得目前觀察狀態
    def _get_obs(self):
        # 狀態清單，從目前步驟開始
        state = [self.current_step]
        # 加電池電量
        state.extend(self.battery_levels)
        # 加是否開始，1或0
        state.extend([1 if s else 0 for s in self.started])
        # 加是否充電中，1或0
        state.extend([1 if c else 0 for c in self.charging])
        # 加剩餘能量
        state.extend(self.remaining_energy)
        # 加目前基載
        state.extend(self.current_scenario)
        # 轉成numpy陣列
        return np.array(state, dtype=np.float32)

    # 環境一步，輸入動作
    def step(self, action):
        # 動作前半是開始動作
        start_actions = action[:self.n_users]
        # 後半是功率動作
        power_actions = action[self.n_users:]
        # 充電功率初始化為0
        charging_powers = [0.0] * self.n_users

        # 開始充電邏輯
        for i in range(self.n_users):
            # 如果沒開始，且還有足夠時間
            if not self.started[i] and self.current_step <= self.n_time_slots - self.duration[i]:
                # 如果動作>0.5，就開始
                if start_actions[i] > 0.5:
                    self.started[i] = True
                    self.start_times[i] = self.current_step
                    self.charging[i] = True

        # 總充電功率初始化
        total_charging_power = 0.0
        for i in range(self.n_users):
            # 如果充電中，且還在持續時間內
            if self.charging[i] and self.current_step < self.start_times[i] + self.duration[i]:
                # 計算功率，動作*最大功率
                power = power_actions[i] * self.max_charging_power
                # 不能超過電池剩餘容量
                power = min(power, self.battery_capacity - self.battery_levels[i])
                # 不能超過剩餘需求
                power = min(power, self.remaining_energy[i] if self.remaining_energy[i] > 0 else 0)
                # 設定功率
                charging_powers[i] = power
                # 更新電池
                self.battery_levels[i] += power
                # 更新剩餘
                self.remaining_energy[i] -= power
                # 加總功率
                total_charging_power += power
            # 如果超過時間，就停止充電
            elif self.charging[i] and self.current_step >= self.start_times[i] + self.duration[i]:
                self.charging[i] = False

        # 總負載 = 基載 + 充電功率
        total_load = self.current_scenario[self.current_step] + total_charging_power
        # 電價 = 0.02*總負載 + 3
        price = 0.02 * total_load + 3
        # 成本 = 電價 * 充電功率
        cost = price * total_charging_power

        # 罰則初始化
        penalty = 0.0
        for i in range(self.n_users):
            # 如果沒開始，且時間不夠了，且還有需求，就罰5000*剩餘
            if not self.started[i] and self.current_step > self.n_time_slots - self.duration[i]:
                if abs(self.remaining_energy[i]) > 1e-6:
                    penalty += 5000 * abs(self.remaining_energy[i])

        # 如果是最後一步，檢查所有剩餘需求，罰則
        if self.current_step == self.n_time_slots - 1:
            for i in range(self.n_users):
                if abs(self.remaining_energy[i]) > 1e-6:
                    penalty += 5000 * abs(self.remaining_energy[i])

        # 獎勵 = -成本 - 罰則
        reward = -cost - penalty
        # 步驟加1
        self.current_step += 1
        # 是否結束，如果步驟>=時間槽
        terminated = self.current_step >= self.n_time_slots
        # truncated 固定False，這個環境沒有額外強制中斷的情況
        truncated = False
        # info 包含充電功率
        info = {'charging_powers': charging_powers}
        # 回傳觀察、獎勵、結束、truncated、info
        return self._get_obs(), float(reward), terminated, truncated, info

    # 顯示環境狀態
    def render(self, mode='human'):
        # 印出目前時間槽
        print(f"Time slot {self.current_step}:")
        for i in range(self.n_users):
            # 狀態是充電中或不充電
            status = "Charging" if self.charging[i] else "Not charging"
            # 印出每個使用者的電池、剩餘、狀態
            print(f"User {i}: Battery={self.battery_levels[i]:.2f}, Remaining={self.remaining_energy[i]:.2f}, Status={status}")


# 建立環境，n_users=5
env = EVChargingEnv(n_users=5)
# 檢查環境有沒有問題
check_env(env)

# 建立 PPO 模型，用 MlpPolicy，verbose=1，學習率0.0003，步驟4096，批次256，用cpu，verbose=1只會輸出最重要的訓練進度資訊
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=4096, batch_size=256, device='cpu')
# 訓練模型，總步驟500000
model.learn(total_timesteps=500000)

# 測試所有情境
# 最壞成本初始化為負無限
worst_cost = float('-inf')
# 最壞情境索引-1
worst_scenario_index = -1
# 最壞排程空清單
worst_schedule = []
# 成本清單
costs = []

# 迴圈每個情境
for scen_idx in range(len(env.base_load_scenarios)):
    # 重置環境，用這個情境
    obs, _ = env.reset(scenario_idx=scen_idx)
    # 總成本0
    total_cost = 0.0
    # 排程清單
    schedule = []
    # 每個時間槽
    for _ in range(env.n_time_slots):
        # 預測動作
        action, _states = model.predict(obs)
        # 一步，取得獎勵等，step函數回傳 觀察、獎勵、結束、truncated、info 包含充電功率
        obs, reward, done, truncated, info = env.step(action)
        # 成本加 -獎勵
        total_cost += -reward
        # 加到排程，包含時間、功率、電池、剩餘、狀態、開始時間
        schedule.append({'time_slot': env.current_step,
                         'charging_powers': info['charging_powers'].copy(),
                         'battery_levels': env.battery_levels.copy(),
                         'remaining_energy': env.remaining_energy.copy(),
                         'charging_status': env.charging.copy(),
                         'start_times': env.start_times.copy()})
        # 如果結束，就break
        if done or truncated:
            break
    # 加成本到清單
    costs.append(total_cost)
    # 如果這個成本 > 最壞，就更新
    if total_cost > worst_cost:
        worst_cost = total_cost
        worst_scenario_index = scen_idx
        worst_schedule = schedule


# 印出最壞成本
print(f"\nTotal charging cost under worst-case scenario = {worst_cost:.2f}")
# 印出最壞情境索引
print(f"Worst-case scenario index: {worst_scenario_index}")


# 計算平均基載，每個時間槽所有情境平均
avg_base_load = [sum(scenario[t] for scenario in env.base_load_scenarios) / env.n_scenarios for t in range(env.n_time_slots)]

# 測試平均基載排程
# 重置環境
obs, _ = env.reset()
# 總成本0
total_cost_avg_base_load = 0.0
# 排程清單
schedule_avg_base_load = []

# 每個時間槽
for t in range(env.n_time_slots):
    # 設目前基礎負載為平均
    env.current_scenario = avg_base_load  # 使用平均基載
    # 預測動作
    action, _states = model.predict(obs)
    # 一步
    obs, reward, done, truncated, info = env.step(action)
    # 加成本
    total_cost_avg_base_load += -reward
    # 加到排程
    schedule_avg_base_load.append({'time_slot': env.current_step,
                                   'charging_powers': info['charging_powers'].copy(),
                                   'battery_levels': env.battery_levels.copy(),
                                   'remaining_energy': env.remaining_energy.copy(),
                                   'charging_status': env.charging.copy(),
                                   'start_times': env.start_times.copy()})
    # 如果結束，break
    if done or truncated:
        break

# 輸出結果
print(f"\n=== Results ===")
# 最壞成本
print(f"Total charging cost under worst-case scenario = {worst_cost:.2f}")
# 平均基載成本
print(f"Total charging cost with average base load = {total_cost_avg_base_load:.2f}")
# 最壞索引
print(f"Worst-case scenario index: {worst_scenario_index}")

# 檢查未滿足需求的使用者
# 未滿足清單
unmet_users = []
# 檢查最壞排程最後一個的剩餘
for i in range(env.n_users):
    final_remaining = worst_schedule[-1]['remaining_energy'][i]
    # 如果剩餘 > 小數，就加到清單
    if abs(final_remaining) > 1e-6:
        unmet_users.append((i, final_remaining))

# 如果有未滿足
if unmet_users:
    # 印警告
    print("\nWARNING: The following users did not meet their energy demands in the worst-case scenario:")
    # 印每個
    for user, remaining in unmet_users:
        print(f"User {user}: Required {env.required_energy[user]:.2f} kWh, Remaining {remaining:.2f} kWh")

# 輸出最壞情境排程
print("\n=== Charging Schedule for Worst-case Scenario ===")
# 每個entry
for entry in worst_schedule:
    # 印時間槽
    print(f"Time slot {entry['time_slot']}:")
    for i in range(env.n_users):
        # 開始時間，如果>=0否則-1
        start_time = entry['start_times'][i] if entry['start_times'][i] >= 0 else -1
        # 印使用者資訊
        print(f"User {i} (required energy: {env.required_energy[i]:.2f} kWh, duration: {env.duration[i]}, start time: {start_time}):")
        # 如果功率 > 容許值
        if entry['charging_powers'][i] > 1e-6:
            # 印功率、電池、剩餘
            print(f"  Power: {entry['charging_powers'][i]:.2f} kW, Battery: {entry['battery_levels'][i]:.2f} kWh, Remaining: {entry['remaining_energy'][i]:.2f} kWh")
        else:
            # 否則沒充電
            print("  No charging activity")
    # 空行
    print()

# 輸出平均基載排程
print("\n=== Charging Schedule for Average Base Load Scenario ===")
# 同上，每個entry
for entry in schedule_avg_base_load:
    print(f"Time slot {entry['time_slot']}:")
    for i in range(env.n_users):
        start_time = entry['start_times'][i] if entry['start_times'][i] >= 0 else -1
        print(f"User {i} (required energy: {env.required_energy[i]:.2f} kWh, duration: {env.duration[i]}, start time: {start_time}):")
        if entry['charging_powers'][i] > 1e-6:
            print(f"  Power: {entry['charging_powers'][i]:.2f} kW, Battery: {entry['battery_levels'][i]:.2f} kWh, Remaining: {entry['remaining_energy'][i]:.2f} kWh")
        else:
            print("  No charging activity")
    print()












