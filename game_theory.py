# 匯入 numpy 套件當作 np，用來處理數字陣列
import numpy as np
# 從 scipy.optimize 匯入 minimize，用來做最佳化計算，找最小值
from scipy.optimize import minimize
# 匯入 random
import random

# 定義 EVChargingGame 類別，這是電動車充電遊戲的模擬環境
class EVChargingGame:
    # 初始化函數，設定使用者數、時間段數量、最大充電功率、電池容量、最大基礎負載參數
    def __init__(self, n_users=25, n_time_slots=24, max_charging_power=50.0, battery_capacity=60.4, max_base_load=220.0):
        # 基本參數
        # 設定使用者數量，這裡設25個使用者
        self.n_users = n_users
        # 設定時間段數量，一天24小時
        self.n_time_slots = n_time_slots
        # 最大充電功率，單位kW
        self.max_charging_power = max_charging_power
        # 電池容量，單位kWh
        self.battery_capacity = battery_capacity
        # 初始電量 (kWh)，設成30.2
        self.initial_soc = 30.2# 初始電量 (kWh)
        # 最大基載功率
        self.max_base_load = max_base_load

        # === 固定使用者需求資料 ===
        # 充電持續時間，每個使用者的充電小時數
        self.duration = [
            4,6,3,5,7,2,8,6,4,3,
            5,7,2,6,4,8,3,5,7,6,
            4,2,5,3,7
        ]
        # 所需充電電量，每個使用者的充電需求kWh
        self.required_energy = [
            12.5,14.0,11.2,13.8,10.5,12.0,15.0,13.2,11.8,12.6,
            14.4,13.7,10.9,12.3,11.5,14.9,13.1,12.8,10.7,13.4,
            11.9,12.2,13.6,14.1,10.8
        ]

        # 修正避免超出容量或時長
        # 迴圈檢查每個使用者
        for i in range(self.n_users):
            # 如果初始電量加需求超過容量，就調整需求減一點點
            if self.initial_soc + self.required_energy[i] > self.battery_capacity:
                self.required_energy[i] = self.battery_capacity - self.initial_soc - 0.1
            # 如果需求超過最大功率乘持續時間，就調整成最大可能
            if self.required_energy[i] > self.max_charging_power * self.duration[i]:
                self.required_energy[i] = self.max_charging_power * self.duration[i]

        # === 50 × 24 的基礎負載情境資料 ===
        # 50組，每組24個時段的基礎負載功率
        hard_base_load = [
    [100, 95, 90, 85, 80, 85, 95, 110, 130, 150, 170, 180,
     190, 200, 210, 220, 210, 200, 180, 160, 140, 130, 120, 110],
    [105, 100, 95, 90, 85, 90, 100, 115, 135, 155, 175, 185,
     195, 205, 215, 210, 200, 190, 170, 150, 135, 125, 115, 108],
    [110, 105, 100, 95, 90, 95, 105, 120, 140, 160, 180, 190,
     200, 210, 220, 215, 205, 195, 175, 155, 138, 128, 118, 112],
    [115, 110, 105, 100, 95, 100, 110, 125, 145, 165, 185, 195,
     205, 215, 210, 205, 198, 188, 172, 152, 136, 126, 116, 111],
    [120, 115, 110, 105, 100, 105, 115, 130, 150, 170, 190, 200,
     210, 220, 215, 210, 200, 190, 174, 154, 138, 128, 118, 113],
    [125, 120, 115, 110, 105, 110, 120, 135, 155, 175, 195, 205,
     215, 210, 205, 200, 195, 185, 170, 150, 134, 124, 114, 109],
    [130, 125, 120, 115, 110, 115, 125, 140, 160, 180, 200, 210,
     215, 205, 200, 195, 190, 180, 165, 148, 132, 122, 112, 108],
    [135, 130, 125, 120, 115, 120, 130, 145, 165, 185, 205, 215,
     210, 200, 195, 190, 185, 175, 160, 145, 130, 120, 110, 105],
    [140, 135, 130, 125, 120, 125, 135, 150, 170, 190, 210, 210,
     205, 195, 190, 185, 180, 170, 158, 143, 128, 118, 108, 103],
    [145, 140, 135, 130, 125, 130, 140, 155, 175, 195, 210, 205,
     200, 190, 185, 180, 175, 165, 155, 140, 126, 116, 106, 101],
    [150, 145, 140, 135, 130, 135, 145, 160, 180, 200, 210, 205,
     195, 185, 180, 175, 170, 160, 150, 138, 124, 114, 104, 100],
    [155, 150, 145, 140, 135, 140, 150, 165, 185, 205, 210, 200,
     190, 180, 175, 170, 165, 158, 148, 136, 122, 112, 102, 98],
    [160, 155, 150, 145, 140, 145, 155, 170, 190, 208, 205, 195,
     185, 178, 172, 168, 162, 155, 145, 134, 120, 110, 100, 95],
    [165, 160, 155, 150, 145, 150, 160, 175, 195, 205, 200, 190,
     182, 175, 170, 165, 160, 152, 142, 132, 118, 108, 98, 94],
    [170, 165, 160, 155, 150, 155, 165, 180, 200, 210, 195, 188,
     180, 172, 168, 162, 158, 150, 140, 130, 116, 106, 96, 92],
    [175, 170, 165, 160, 155, 160, 170, 185, 205, 208, 192, 185,
     178, 170, 165, 160, 155, 148, 138, 128, 114, 104, 94, 90],
    [180, 175, 170, 165, 160, 165, 175, 190, 208, 205, 190, 182,
     175, 168, 162, 158, 152, 146, 136, 126, 112, 102, 92, 88],
    [185, 180, 175, 170, 165, 170, 180, 195, 210, 202, 188, 180,
     172, 165, 160, 155, 150, 144, 134, 124, 110, 100, 90, 86],
    [190, 185, 180, 175, 170, 175, 185, 200, 210, 200, 185, 178,
     170, 162, 158, 152, 148, 142, 132, 122, 108, 98, 88, 84],
    [195, 190, 185, 180, 175, 180, 190, 205, 208, 198, 182, 175,
     168, 160, 155, 150, 145, 140, 130, 120, 106, 96, 86, 82],
    [200, 195, 190, 185, 180, 185, 195, 208, 205, 195, 180, 172,
     165, 158, 152, 148, 142, 138, 128, 118, 104, 94, 84, 80],
    [205, 200, 195, 190, 185, 190, 200, 210, 202, 192, 178, 170,
     162, 155, 150, 145, 140, 135, 125, 115, 102, 92, 82, 80],
    [210, 205, 200, 195, 190, 195, 205, 210, 200, 190, 175, 168,
     160, 152, 148, 142, 138, 132, 122, 112, 100, 90, 82, 80],
    [208, 203, 198, 193, 188, 193, 203, 208, 198, 188, 173, 166,
     158, 150, 145, 140, 136, 130, 120, 110, 98, 88, 82, 80],
    [206, 201, 196, 191, 186, 191, 201, 206, 196, 186, 171, 164,
     156, 148, 143, 138, 134, 128, 118, 108, 96, 86, 82, 80],
    [204, 199, 194, 189, 184, 189, 199, 204, 194, 184, 169, 162,
     154, 146, 141, 136, 132, 126, 116, 106, 94, 84, 82, 80],
    [202, 197, 192, 187, 182, 187, 197, 202, 192, 182, 167, 160,
     152, 144, 139, 134, 130, 124, 114, 104, 92, 84, 82, 80],
    [200, 195, 190, 185, 180, 185, 195, 200, 190, 180, 165, 158,
     150, 142, 137, 132, 128, 122, 112, 102, 90, 84, 82, 80],
    [198, 193, 188, 183, 178, 183, 193, 198, 188, 178, 163, 156,
     148, 140, 135, 130, 126, 120, 110, 100, 88, 84, 82, 80],
    [196, 191, 186, 181, 176, 181, 191, 196, 186, 176, 161, 154,
     146, 138, 133, 128, 124, 118, 108, 98, 86, 84, 82, 80],
    [194, 189, 184, 179, 174, 179, 189, 194, 184, 174, 159, 152,
     144, 136, 131, 126, 122, 116, 106, 96, 84, 84, 82, 80],
    [192, 187, 182, 177, 172, 177, 187, 192, 182, 172, 157, 150,
     142, 134, 129, 124, 120, 114, 104, 94, 82, 84, 82, 80],
    [190, 185, 180, 175, 170, 175, 185, 190, 180, 170, 155, 148,
     140, 132, 127, 122, 118, 112, 102, 92, 82, 84, 82, 80],
    [188, 183, 178, 173, 168, 173, 183, 188, 178, 168, 153, 146,
     138, 130, 125, 120, 116, 110, 100, 90, 82, 84, 82, 80],
    [186, 181, 176, 171, 166, 171, 181, 186, 176, 166, 151, 144,
     136, 128, 123, 118, 114, 108, 98, 88, 82, 84, 82, 80],
    [184, 179, 174, 169, 164, 169, 179, 184, 174, 164, 149, 142,
     134, 126, 121, 116, 112, 106, 96, 86, 82, 84, 82, 80],
    [182, 177, 172, 167, 162, 167, 177, 182, 172, 162, 147, 140,
     132, 124, 119, 114, 110, 104, 94, 84, 82, 84, 82, 80],
    [180, 175, 170, 165, 160, 165, 175, 180, 170, 160, 145, 138,
     130, 122, 117, 112, 108, 102, 92, 82, 82, 84, 82, 80],
    [178, 173, 168, 163, 158, 163, 173, 178, 168, 158, 143, 136,
     128, 120, 115, 110, 106, 100, 90, 80, 82, 84, 82, 80],
    [176, 171, 166, 161, 156, 161, 171, 176, 166, 156, 141, 134,
     126, 118, 113, 108, 104, 98, 88, 80, 82, 84, 82, 80],
    [174, 169, 164, 159, 154, 159, 169, 174, 164, 154, 139, 132,
     124, 116, 111, 106, 102, 96, 86, 80, 82, 84, 82, 80],
    [172, 167, 162, 157, 152, 157, 167, 172, 162, 152, 137, 130,
     122, 114, 109, 104, 100, 94, 84, 80, 82, 84, 82, 80],
    [170, 165, 160, 155, 150, 155, 165, 170, 160, 150, 135, 128,
     120, 112, 107, 102, 98, 92, 82, 80, 82, 84, 82, 80],
    [168, 163, 158, 153, 148, 153, 163, 168, 158, 148, 133, 126,
     118, 110, 105, 100, 96, 90, 80, 80, 82, 84, 82, 80],
    [166, 161, 156, 151, 146, 151, 161, 166, 156, 146, 131, 124,
     116, 108, 103, 98, 94, 88, 80, 80, 82, 84, 82, 80],
    [164, 159, 154, 149, 144, 149, 159, 164, 154, 144, 129, 122,
     114, 106, 101, 96, 92, 86, 80, 80, 82, 84, 82, 80],
    [162, 157, 152, 147, 142, 147, 157, 162, 152, 142, 127, 120,
     112, 104, 99, 94, 90, 84, 80, 80, 82, 84, 82, 80],
    [160, 155, 150, 145, 140, 145, 155, 160, 150, 140, 125, 118,
     110, 102, 97, 92, 88, 82, 80, 80, 82, 84, 82, 80],
    [158, 153, 148, 143, 138, 143, 153, 158, 148, 138, 123, 116,
     108, 100, 95, 90, 86, 80, 80, 80, 82, 84, 82, 80],
    [156, 151, 146, 141, 136, 141, 151, 156, 146, 136, 121, 114,
     106, 98, 93, 88, 84, 80, 80, 80, 82, 84, 82, 80],
]

        # 情境數量，50個
        self.n_scenarios = 50
        # 基載情境清單
        self.base_load_scenarios = hard_base_load
        # 目前情境，用第一個
        self.current_scenario = self.base_load_scenarios[0]

        # 初始化開始充電時間與功率配置
        # 每個使用者的隨機開始時間
        self.start_times = [
            random.randint(0, self.n_time_slots - self.duration[i]) for i in range(self.n_users)
        ]
        # 每個使用者的功率配置，均勻分配
        self.power_profiles = [
            [self.required_energy[i] / self.duration[i]] * self.duration[i] for i in range(self.n_users)
        ]

    # 計算所有使用者在每個時段的總充電電量
    def compute_aggregate_charging(self, start_times=None, power_profiles=None, exclude_user=None):
        # 如果沒給開始時間，就用自己的
        if start_times is None:
            start_times = self.start_times
        # 如果沒給功率配置，就用自己的
        if power_profiles is None:
            power_profiles = self.power_profiles
        # 初始化總充電負載為 0 的清單，長度是時間槽
        agg_charging = [0.0] * self.n_time_slots  # 初始化總充電負載為 0
        # 迴圈每個使用者
        for i in range(self.n_users):
            # 如果是排除的使用者，就跳過
            if i == exclude_user:  # 排除某位使用者（例如在做 best response 時）
                continue
            # 取得開始時間
            start_time = start_times[i]
            # 迴圈持續時間
            for k in range(self.duration[i]):
                # 計算時段 t
                t = start_time + k
                # 如果 t 在範圍內
                if t < self.n_time_slots:
                    # 加到總充電
                    agg_charging[t] += power_profiles[i][k]  # 累加各使用者在時段 t 的充電量
        # 回傳總充電清單
        return agg_charging

    # 對某位使用者，根據指定開始時間，最佳化其功率配置以最小化成本
    def optimize_power_profile(self, user_idx, start_time, start_times, power_profiles):
        # 取得持續時間
        duration = self.duration[user_idx]
        # 計算其他人的總充電，排除自己
        agg_charging = self.compute_aggregate_charging(start_times, power_profiles, exclude_user=user_idx)
        # 儲存每個時段的基礎負載 + 其他車的充電量
        constant_t = []  # 儲存每個時段的基礎負載 + 其他車的充電量
        # 迴圈持續時間
        for k in range(duration):
            # 計算 t
            t = start_time + k
            # 如果 t 在範圍內
            if t < self.n_time_slots:
                # 加到 constant_t
                constant_t.append(self.current_scenario[t] + agg_charging[t])
            else:
                # 超出範圍，回傳無效
                return None, float('inf')  # 時段超出範圍，回傳無效結果

        # 定義成本函數：與電力使用量和電價線性相關
        # P 代表 對某一位使用者在每個時段的充電功率 (Power profile)，它是一個向量 (list / array)，長度等於 duration（也就是使用者的充電持續時間）
        def objective(P):
            # 成本初始化0
            cost = 0.0
            # 迴圈每個 k
            for k in range(duration):
                # 取得 P_k，P[k] 就是第 k 個時段的功率
                P_k = P[k]
                # 加成本
                cost +=  (0.02 * constant_t[k] + 3) * P_k
            # 回傳成本
            return cost

        # 限制條件：充電總量必須等於需求
        # 限制條件強迫 sum(P) == 使用者所需充電量。也就是不管怎麼分配，最後總能量必須等於需求。
        constraints = [{'type': 'eq', 'fun': lambda P: sum(P) - self.required_energy[user_idx]}]
        # 每個時段的功率限制，從0到最大
        bounds = [(0, self.max_charging_power)] * duration  # 每個時段的功率限制
        # 初始猜測：均勻分配
        P0 = [self.required_energy[user_idx] / duration] * duration  # 初始猜測：均勻分配
        # 用 minimize 最佳化，用 SLSQP 方法
        result = minimize(objective, P0, method='SLSQP', bounds=bounds, constraints=constraints)
        # 如果成功
        if result.success:
            # 回傳最佳功率和成本
            return result.x.tolist(), result.fun  # 回傳最佳功率配置與對應成本
        # 失敗，回傳初始和無限成本
        return P0, float('inf')  # 若失敗，回傳初始猜測與無限成本

    # 找出某位使用者的最佳反應（最佳開始時間與功率配置）
    def find_best_response(self, user_idx, start_times, power_profiles):
        # 最佳開始時間初始化 None
        best_start_time = None
        # 最佳功率初始化 None
        best_power_profile = None
        # 最佳成本初始化無限
        best_cost = float('inf')
        # 迴圈可能開始時間
        for start_time in range(self.n_time_slots - self.duration[user_idx] + 1):
            # 最佳化功率
            power_profile, cost = self.optimize_power_profile(user_idx, start_time, start_times, power_profiles)
            # 如果成本更好
            if cost < best_cost:
                # 更新最佳
                best_cost = cost
                best_start_time = start_time
                best_power_profile = power_profile
        # 回傳最佳
        return best_start_time, best_power_profile, best_cost

    # 執行多回合的最佳反應直到收斂（模擬納許均衡）
    def find_nash_equilibrium(self, max_iterations=500, tolerance=1e-3):
        # 複製開始時間
        start_times = self.start_times.copy()
        # 複製功率配置
        power_profiles = [profile.copy() for profile in self.power_profiles]
        # 迭代計數0
        iteration_count = 0
        # 迴圈最大迭代
        for iteration in range(max_iterations):
            # 計數加1
            iteration_count += 1
            # 是否改變初始化 False
            changed = False
            # 使用者索引清單
            user_indices = list(range(self.n_users))
            # 隨機打亂順序
            random.shuffle(user_indices)  # 隨機順序更新使用者
            # 迴圈每個使用者
            for i in user_indices:
                # 找最佳反應
                best_start_time, best_power_profile, best_cost = self.find_best_response(i, start_times, power_profiles)
                # 若策略有變化（開始時間或功率配置有差異），則更新
                if (best_start_time != start_times[i] or 
                    sum((a - b)**2 for a, b in zip(best_power_profile, power_profiles[i])) > tolerance**2):
                    # 更新開始時間
                    start_times[i] = best_start_time
                    # 更新功率
                    power_profiles[i] = best_power_profile
                    # 設 changed True
                    changed = True
            # 如果沒改變
            if not changed:
                # 印收斂訊息
                print(f"Converged after {iteration_count} iterations")  # 成功收斂
                # 更新自己
                self.start_times = start_times
                self.power_profiles = power_profiles
                # 回傳開始時間和計數
                return start_times, iteration_count
        # 未收斂印訊息
        print(f"Did not converge within {max_iterations} iterations")  # 未收斂
        # 更新自己
        self.start_times = start_times
        self.power_profiles = power_profiles
        # 回傳 -1 表示未收斂
        return start_times, -1

    # 計算總成本：根據目前充電排程與價格模型計算所有使用者的總成本
    def compute_total_cost(self, start_times, power_profiles):
        # 計算總充電
        agg_charging = self.compute_aggregate_charging(start_times, power_profiles)
        # 總成本0
        total_cost = 0.0
        # 迴圈每個使用者
        for i in range(self.n_users):
            # 使用者成本0
            user_cost = 0.0
            # 取得開始時間
            start_time = start_times[i]
            # 迴圈持續時間
            for k in range(self.duration[i]):
                # 計算 t
                t = start_time + k
                # 如果 t 在範圍
                if t < self.n_time_slots:
                    # 取得功率
                    P_i_t = power_profiles[i][k]
                    # 總負載
                    total_load = self.current_scenario[t] + agg_charging[t]
                    # 電價
                    price = 0.02 * total_load + 3  # 電價模型
                    # 加使用者成本
                    user_cost += price * P_i_t
            # 加到總成本
            total_cost += user_cost
        # 回傳總成本
        return total_cost

    # 檢查每位使用者是否滿足充電需求
    def verify_energy_requirements(self, start_times, power_profiles):
        # 未滿足清單
        unmet_users = []
        # 迴圈每個使用者
        for i in range(self.n_users):
            # 總能量
            total_energy = sum(power_profiles[i])
            # 如果差異大
            if abs(total_energy - self.required_energy[i]) > 1e-6:
                # 加到清單
                unmet_users.append((i, self.required_energy[i] - total_energy))
        # 回傳未滿足
        return unmet_users

    # 執行完整模擬：遍歷每個場景，找出最壞成本和平均成本
    def run_simulation(self):
        # 迭代計數清單
        iteration_counts = []
        # 最壞成本負無限
        worst_cost = float('-inf')
        # 最壞索引 -1
        worst_scenario_index = -1
        # 最壞開始時間 None
        worst_start_times = None
        # 最壞功率 None
        worst_power_profiles = None
        # 成本清單
        costs = []

        # 迴圈每個情境
        for scen_idx in range(self.n_scenarios):
            # 設目前情境
            self.current_scenario = self.base_load_scenarios[scen_idx]
            # 隨機開始時間
            self.start_times = [
                random.randint(0, self.n_time_slots - self.duration[i]) for i in range(self.n_users)
            ]
            # 均勻功率
            self.power_profiles = [
                [self.required_energy[i] / self.duration[i]] * self.duration[i] for i in range(self.n_users)
            ]
            # 找納許均衡
            start_times, iter_count = self.find_nash_equilibrium(max_iterations=500)
            # 加迭代計數
            iteration_counts.append(iter_count)
            # 計算總成本
            total_cost = self.compute_total_cost(start_times, self.power_profiles)
            # 加到成本
            costs.append(total_cost)
            # 如果 > 最壞
            if total_cost > worst_cost:
                # 更新最壞
                worst_cost = total_cost
                worst_scenario_index = scen_idx
                worst_start_times = start_times.copy()
                worst_power_profiles = [profile.copy() for profile in self.power_profiles]

        # 計算平均成本與平均基礎負載場景下的結果
        # 平均成本
        average_cost = sum(costs) / len(costs)
        # 平均基載，每時段平均
        avg_base_load = [
            sum(scenario[t] for scenario in self.base_load_scenarios) / self.n_scenarios
            for t in range(self.n_time_slots)
        ]
        # 設目前為平均
        self.current_scenario = avg_base_load
        # 隨機開始
        self.start_times = [
            random.randint(0, self.n_time_slots - self.duration[i]) for i in range(self.n_users)
        ]
        # 均勻功率
        self.power_profiles = [
            [self.required_energy[i] / self.duration[i]] * self.duration[i] for i in range(self.n_users)
        ]
        # 找均衡
        avg_start_times, iter_count = self.find_nash_equilibrium(max_iterations=500)
        # 加計數
        iteration_counts.append(iter_count)
        # 計算平均基載成本
        total_cost_avg_base_load = self.compute_total_cost(avg_start_times, self.power_profiles)

        # 印出結果
        print(f"\n=== Results ===")
        # 最壞成本
        print(f"Total charging cost under worst-case scenario = {worst_cost:.2f}")
        # 平均基載成本
        print(f"Total charging cost with average base load = {total_cost_avg_base_load:.2f}")
        # 最壞索引
        print(f"Worst-case scenario index: {worst_scenario_index}")

        # 總迭代
        total_iterations = sum(count for count in iteration_counts if count > 0)
        # 未收斂數
        unconverged_scenarios = sum(1 for count in iteration_counts if count == -1)
        # 印迭代統計
        print(f"\n=== 迭代次數統計 ===")
        # 總迭代
        print(f"所有場景的總迭代次數: {total_iterations}")
        # 未收斂數
        print(f"未收斂的場景數量: {unconverged_scenarios}")

        # 檢查最壞場景下有沒有使用者未滿足需求
        # 驗證需求
        unmet_users = self.verify_energy_requirements(worst_start_times, worst_power_profiles)
        # 如果有
        if unmet_users:
            # 印警告
            print("\nWARNING: The following users did not meet their energy demands in the worst-case scenario:")
            # 迴圈印每個
            for user, remaining in unmet_users:
                print(f"User {user}: Required {self.required_energy[user]:.2f} kWh, Remaining {remaining:.2f} kWh")

# 建立一個遊戲實例並執行模擬
# 建立遊戲，
game = EVChargingGame(n_users=25)
# 執行模擬
game.run_simulation()





