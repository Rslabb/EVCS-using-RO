#include <gurobi_c++.h>      // 引入 Gurobi C++ API 標頭檔
#include <vector>           // 引入向量容器支援
#include <random>           // 引入隨機數產生器支援
#include <ctime>            // 引入時間函式
#include <cstdlib>          // 引入標準函式庫
#include <fstream>          // 引入檔案輸入輸出支援
#include <iostream>         // 引入輸出入串流支援
#include <sstream>          // 引入字串串流支援
#include <time.h>           // 引入舊版時間函式
#include <algorithm>        // 引入演算法支援（如 sort）
#include <iomanip>          // 引入格式化輸出支援
using namespace std;        // 使用標準命名空間

// 幫助函式：設定變數或約束名稱 (含使用者編號 i 和時段 t)
string setVarOrConstrName(string str, int i, int t) {
    ostringstream vname;    // 建立字串串流
    vname << str << "[" << i << "]" << "[" << t << "]";  // 組合名稱
    return vname.str();     // 回傳字串
}

// 幫助函式：設定變數或約束名稱 (含使用者編號 i)
string setVarOrConstrName(string str, int i) {
    ostringstream vname;    
    vname << str << "[" << i << "]";   // 組合名稱
    return vname.str();     
}

// 幫助函式：設定變數或約束名稱 (含使用者編號 i、j 和時段 t)
string setVarOrConstrName(string str, int i, int j, int t) {
    ostringstream vname;    
    vname << str << "[" << i << "]" << "[" << j << "]" << "[" << t << "]";  // 組合名稱
    return vname.str();     
}

int main(int argc, char *argv[])
{
    // 記錄程式開始時間
    clock_t t1, t2;          
    t1 = clock();            // 讀取初始時間

    // 初始化常數
    const double maxChargingPower = 50.0;              // 最大充電功率 (kW)
    const double bignumber = maxChargingPower;         // 大 M 方法用
    const double maxBaseLoad = 220.0;                  // 最大基載 (kW)
    const int nTimeSlots = 24;                         // 時段數
    const double batteryCapacity = 60.4;               // 電池容量 (kWh)
    const double initialSOC = batteryCapacity / 2;     // 初始電量 (一半容量)
    int worstScenIndex = 0;                            // 最壞情境索引
    const int maxScenarios = 100;                      // 設定情境上限
    const double epsilon = 1e-10;                      // 精度門檻

        // 產生使用者和情境數量
    int nUsers = 5;                                   // 使用者數
    int nScenarios = 50;                              // 情境數 
    double maxLoad = maxBaseLoad + nUsers * maxChargingPower; // 最大總負載
    int *duration = new int[nUsers];                  // 每位使用者充電時長
    double *requiredEnergy = new double[nUsers];      // 每位使用者需求能量
    cout << "Users: " << nUsers << ", Scenarios: " << nScenarios << endl; // 列印參數

        // 使用硬編碼的基載情境資料 (範圍 80–220)
    double base_load[50][24] ={
            {200,195,190,185,180,185,195,200,190,180,165,158,150,142,137,132,128,122,112,106,108,110,108,106},
            {198,193,188,183,178,183,193,198,188,178,163,156,148,140,135,130,126,120,110,106,108,110,108,106},
            {196,191,186,181,176,181,191,196,186,176,161,154,146,138,133,128,124,118,108,106,108,110,108,106},
            {194,189,184,179,174,179,189,194,184,174,159,152,144,136,131,126,122,116,106,106,108,110,108,106},
            {192,187,182,177,172,177,187,192,182,172,157,150,142,134,129,124,120,114,104,106,108,110,108,106},
            {190,185,180,175,170,175,185,190,180,170,155,148,140,132,127,122,118,112,102,106,108,110,108,106},
            {188,183,178,173,168,173,183,188,178,168,153,146,138,130,125,120,116,110,100,106,108,110,108,106},
            {186,181,176,171,166,171,181,186,176,166,151,144,136,128,123,118,114,108,98,106,108,110,108,106},
            {184,179,174,169,164,169,179,184,174,164,149,142,134,126,121,116,112,106,96,106,108,110,108,106},
            {182,177,172,167,162,167,177,182,172,162,147,140,132,124,119,114,110,104,94,106,108,110,108,106},
            {180,175,170,165,160,165,175,180,170,160,145,138,130,122,117,112,108,102,92,106,108,110,108,106},
            {178,173,168,163,158,163,173,178,168,158,143,136,128,120,115,110,106,100,90,106,108,110,108,106},
            {176,171,166,161,156,161,171,176,166,156,141,134,126,118,113,108,104,98,88,106,108,110,108,106},
            {174,169,164,159,154,159,169,174,164,154,139,132,124,116,111,106,102,96,86,106,108,110,108,106},
            {172,167,162,157,152,157,167,172,162,152,137,130,122,114,109,104,100,94,84,106,108,110,108,106},
            {170,165,160,155,150,155,165,170,160,150,135,128,120,112,107,102,98,92,82,106,108,110,108,106},
            {168,163,158,153,148,153,163,168,158,148,133,126,118,110,105,100,96,90,80,106,108,110,108,106},
            {166,161,156,151,146,151,161,166,156,146,131,124,116,108,103,98,94,88,80,106,108,110,108,106},
            {164,159,154,149,144,149,159,164,154,144,129,122,114,106,101,96,92,86,80,106,108,110,108,106},
            {162,157,152,147,142,147,157,162,152,142,127,120,112,104,99,94,90,84,80,106,108,110,108,106},
            {160,155,150,145,140,145,155,160,150,140,125,118,110,102,97,92,88,82,80,106,108,110,108,106},
            {158,153,148,143,138,143,153,158,148,138,123,116,108,100,95,90,86,80,80,106,108,110,108,106},
            {156,151,146,141,136,141,151,156,146,136,121,114,106,98,93,88,84,80,80,106,108,110,108,106},
            {154,149,144,139,134,139,149,154,144,134,119,112,104,96,91,86,82,80,80,106,108,110,108,106},
            {152,147,142,137,132,137,147,152,142,132,117,110,102,94,89,84,80,80,80,106,108,110,108,106},
            {150,145,140,135,130,135,145,150,140,130,115,108,100,92,87,82,80,80,80,106,108,110,108,106},
            {148,143,138,133,128,133,143,148,138,128,113,106,98,90,85,80,80,80,80,106,108,110,108,106},
            {146,141,136,131,126,131,141,146,136,126,111,104,96,88,83,80,80,80,80,106,108,110,108,106},
            {144,139,134,129,124,129,139,144,134,124,109,102,94,86,81,80,80,80,80,106,108,110,108,106},
            {142,137,132,127,122,127,137,142,132,122,107,100,92,84,79,80,80,80,80,106,108,110,108,106},
            {140,135,130,125,120,125,135,140,130,120,105,98,90,82,77,80,80,80,80,106,108,110,108,106},
            {138,133,128,123,118,123,133,138,128,118,103,96,88,80,75,80,80,80,80,106,108,110,108,106},
            {136,131,126,121,116,121,131,136,126,116,101,94,86,78,73,80,80,80,80,106,108,110,108,106},
            {134,129,124,119,114,119,129,134,124,114,99,92,84,76,71,80,80,80,80,106,108,110,108,106},
            {132,127,122,117,112,117,127,132,122,112,97,90,82,74,69,80,80,80,80,106,108,110,108,106},
            {130,125,120,115,110,115,125,130,120,110,95,88,80,72,67,80,80,80,80,106,108,110,108,106},
            {128,123,118,113,108,113,123,128,118,108,93,86,78,70,65,80,80,80,80,106,108,110,108,106},
            {126,121,116,111,106,111,121,126,116,106,91,84,76,68,63,80,80,80,80,106,108,110,108,106},
            {124,119,114,109,104,109,119,124,114,104,89,82,74,66,61,80,80,80,80,106,108,110,108,106},
            {122,117,112,107,102,107,117,122,112,102,87,80,72,64,59,80,80,80,80,106,108,110,108,106},
            {120,115,110,105,100,105,115,120,110,100,85,78,70,62,57,80,80,80,80,106,108,110,108,106},
            {118,113,108,103,98,103,113,118,108,98,83,76,68,60,55,80,80,80,80,106,108,110,108,106},
            {116,111,106,101,96,101,111,116,106,96,81,74,66,58,53,80,80,80,80,106,108,110,108,106},
            {114,109,104,99,94,99,109,114,104,94,79,72,64,56,51,80,80,80,80,106,108,110,108,106},
            {112,107,102,97,92,97,107,112,102,92,77,70,62,54,49,80,80,80,80,106,108,110,108,106},
            {110,105,100,95,90,95,105,110,100,90,75,68,60,52,47,80,80,80,80,106,108,110,108,106}
        }


    double **base_load_set = new double *[maxScenarios];
    for (int s = 0; s < maxScenarios; s++) {
        base_load_set[s] = new double[nTimeSlots];
        for (int t = 0; t < nTimeSlots; t++) {
            base_load_set[s][t] = base_load[s][t];
        }
    }

   // === 使用者需求資料 (25人固定) ===
    int hard_duration[25] = {
        4,6,3,5,7,2,8,6,4,3,
        5,7,2,6,4,8,3,5,7,6,
        4,2,5,3,7
    };

    double hard_requiredEnergy[25] = {
        12.5,14.0,11.2,13.8,10.5,12.0,15.0,13.2,11.8,12.6,
        14.4,13.7,10.9,12.3,11.5,14.9,13.1,12.8,10.7,13.4,
        11.9,12.2,13.6,14.1,10.8
    };

    for (int i = 0; i < nUsers; i++) {
        duration[i] = hard_duration[i];
        requiredEnergy[i] = hard_requiredEnergy[i];
        if (initialSOC + requiredEnergy[i] > batteryCapacity) {
            requiredEnergy[i] = batteryCapacity - initialSOC - 0.1;
        }
    }


    // 列印問題規模
    cout << "Total users: " << nUsers << endl;        
    cout << "Total time slots: " << nTimeSlots << endl;
    cout << "Total scenarios: " << nScenarios << endl;

    try {
        // 建立 Gurobi 環境與模型
        GRBEnv env = GRBEnv();                        
        GRBModel master_model = GRBModel(env);         // 建立主問題模型
        master_model.set("NumericFocus", "3");         // 提升數值精度
        master_model.set(GRB_IntParam_Threads, 0);     // 自動設定執行緒數
        master_model.set(GRB_DoubleParam_MIPGap, 0.05);// 設定允許 Gap
        master_model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);// 可行性公差

        // 定義決策變數
        GRBVar **chargingPower = new GRBVar *[nUsers]; // 充電功率
        GRBVar **batteryLevel = new GRBVar *[nUsers];  // 電池電量
        GRBVar **Sigma = new GRBVar *[nUsers];         // 啟動指示器
        GRBVar **I = new GRBVar *[nUsers];             // 激活指示器
        for (int i = 0; i < nUsers; i++) {z/
            chargingPower[i] = new GRBVar[nTimeSlots]; 
            batteryLevel[i] = new GRBVar[nTimeSlots + 1]; 
            Sigma[i] = new GRBVar[nTimeSlots];        
            I[i] = new GRBVar[nTimeSlots];           
            for (int t = 0; t < nTimeSlots; t++) {
                // 連續變數：充電功率
                chargingPower[i][t] = master_model.addVar(0.0, maxChargingPower, 0.0, GRB_CONTINUOUS,
                    setVarOrConstrName("ChargingPower", i, t));
                // 二元變數：開始充電指示
                Sigma[i][t] = master_model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                    setVarOrConstrName("Sigma", i, 0, t));
                // 二元變數：活躍狀態指示
                I[i][t] = master_model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                    setVarOrConstrName("I", i, 0, t));
            }
            for (int t = 0; t <= nTimeSlots; t++) {
                // 連續變數：電池電量
                batteryLevel[i][t] = master_model.addVar(0.0, batteryCapacity, 0.0, GRB_CONTINUOUS,
                    setVarOrConstrName("BatteryLevel", i, t));
            }
        }
        // 極值變數：最壞情境成本
        GRBVar eta = master_model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "eta");
        master_model.update();  // 更新模型

        // 電池動態與總能量約束
        for (int i = 0; i < nUsers; i++) {
            // 初始電量設定
            master_model.addConstr(batteryLevel[i][0] == initialSOC, setVarOrConstrName("InitialSOC", i));
            for (int t = 1; t <= nTimeSlots; t++) {
                // 電量隨充電功率累積變化
                master_model.addConstr(batteryLevel[i][t] == batteryLevel[i][t - 1] + chargingPower[i][t - 1],
                    setVarOrConstrName("BatteryUpdate", i, t));
            }
            // 總充電能量等於需求
            GRBLinExpr totalEnergy = 0;
            for (int t = 0; t < nTimeSlots; t++) totalEnergy += chargingPower[i][t];
            master_model.addConstr(totalEnergy == requiredEnergy[i], setVarOrConstrName("TotalEnergy", i));
        }

        // 開始時段只能選一次約束
        for (int i = 0; i < nUsers; i++) {
            GRBLinExpr startConstr = 0;
            // 無法在無效時段開始
            for (int t = nTimeSlots - duration[i] + 1; t < nTimeSlots; t++) {
                master_model.addConstr(Sigma[i][t] == 0, setVarOrConstrName("StartConstr", i, 0, t));
            }
            // 必須且只能選一次開始時段
            for (int t = 0; t <= nTimeSlots - duration[i]; t++) startConstr += Sigma[i][t];
            master_model.addConstr(startConstr == 1, setVarOrConstrName("StartOnce", i));
        }

        // 活躍指示器
        for (int i = 0; i < nUsers; i++) {
            for (int t = 0; t < nTimeSlots; t++) {
                GRBLinExpr activeConstr = 0;
                // 如果在有效區間內即視為活躍
                for (int start = max(0, t - duration[i] + 1); start <= t; start++)
                    activeConstr += Sigma[i][start];
                master_model.addConstr(I[i][t] == activeConstr, setVarOrConstrName("ActiveConstr", i, 0, t));
            }
        }
        master_model.update();  // 更新模型

        // C&CG 主迴圈初值
        double Upper_bound = 99999999999.0;   
        double Lower_bound = 0.0;            
        double gap = 10;                     
        double best_Upper_bound = GRB_INFINITY;//每次迭代更新後的上界值 
        vector<double> worst_base_load;     
        vector<vector<GRBVar>> Load;       
        int iter = 0;                       

        // 迴圈條件：上下界差距大於閾值
        while (best_Upper_bound - Lower_bound >= gap) {
            cout << "\nIteration " << iter << " Master start\n";  // 列印迭代訊息
            Load.resize(iter + 1);                     // 新增當前迭代的 Load 變數空間
            Load.at(iter).resize(nTimeSlots);         
            for (int t = 0; t < nTimeSlots; t++) {
                // 定義每個時段的 Load 變數
                Load.at(iter).at(t) = master_model.addVar(0.0, maxLoad, 0.0, GRB_CONTINUOUS,
                    setVarOrConstrName("Load", iter, t));
            }
            master_model.update();  // 更新模型

            // 總負載約束 (加上基載)
            for (int t = 0; t < nTimeSlots; t++) {
                GRBLinExpr TotalLoad_RHS = 0;
                for (int i = 0; i < nUsers; i++) TotalLoad_RHS += chargingPower[i][t];
                // 第一次用第一個基載情境，之後用最壞基載
                if (iter == 0) TotalLoad_RHS += base_load_set[0][t];
                else TotalLoad_RHS += worst_base_load[t];
                master_model.addConstr(Load.at(iter).at(t) == TotalLoad_RHS,
                    setVarOrConstrName("Master_Load_Constr", iter, t));
            }
            master_model.update();  // 更新模型

            // 目標函數的準備：二次式約束
            GRBQuadExpr obj_constr = 0;
            for (int t = 0; t < nTimeSlots; t++) {
                GRBLinExpr price = 0.02 * Load.at(iter).at(t) + 3;  // 價格函數
                for (int i = 0; i < nUsers; i++) obj_constr += price * chargingPower[i][t];
            }
            master_model.addQConstr(eta >= obj_constr, setVarOrConstrName("Master_obj", iter));
            master_model.update();  // 更新模型

            // 設定並求解主問題
            master_model.setObjective(eta, GRB_MINIMIZE);  
            master_model.write(to_string(iter) + "master.lp"); // 輸出 LP 檔
            master_model.optimize();  // 最小化
            Lower_bound = master_model.get(GRB_DoubleAttr_ObjVal); // 更新下界

            // 子問題：針對每個基載情境計算成本
            double max_cost = 0;
            worstScenIndex = 0;
            for (int s = 0; s < nScenarios; s++) {
                double cost = 0;
                for (int t = 0; t < nTimeSlots; t++) {
                    double total_load = base_load_set[s][t];
                    for (int i = 0; i < nUsers; i++)
                        total_load += chargingPower[i][t].get(GRB_DoubleAttr_X);
                    double price = 0.02 * total_load + 2;
                    for (int i = 0; i < nUsers; i++)
                        cost += price * chargingPower[i][t].get(GRB_DoubleAttr_X);
                }
                if (cost > max_cost) {  // 找出最大成本
                    max_cost = cost;
                    worstScenIndex = s;
                }
            }
            Upper_bound = max_cost;  
            best_Upper_bound = min(best_Upper_bound, max_cost); // 更新上界

            // 更新最壞基載情境
            worst_base_load.clear();
            for (int t = 0; t < nTimeSlots; t++)
                worst_base_load.push_back(base_load_set[worstScenIndex][t]);

            cout << "iter=" << iter << ", max_cost: " << max_cost
                 << ", best_Upper_bound: " << best_Upper_bound
                 << ", Lower_bound: " << Lower_bound << endl;
            iter++;  // 進入下一迭代
        }

        // 記錄結束時間並輸出
        t2 = clock();
        printf("Total time: %lf\n", (t2 - t1) / (double)(CLOCKS_PER_SEC)); // 輸出總耗時

        // 檢查求解狀態
        int status = master_model.get(GRB_IntAttr_Status);
        if (status == GRB_OPTIMAL) {
            double ObjValue = master_model.get(GRB_DoubleAttr_ObjVal); // 最佳目標值
            cout << "Total charging cost = " << ObjValue << endl;

            // 平均情境下的充電成本計算
            double* avg_base_load = new double[nTimeSlots];
            for (int t = 0; t < nTimeSlots; t++) {
                avg_base_load[t] = 0.0;
                for (int s = 0; s < nScenarios; s++)
                    avg_base_load[t] += base_load_set[s][t];
                avg_base_load[t] /= nScenarios; // 計算平均基載
            }

            double average_cost = 0.0;
            for (int t = 0; t < nTimeSlots; t++) {
                double total_load = avg_base_load[t];
                double sum_demand = 0.0;
                for (int i = 0; i < nUsers; i++) {
                    double demand = chargingPower[i][t].get(GRB_DoubleAttr_X);
                    total_load += demand;
                    sum_demand += demand;
                }
                double price = 0.02 * total_load + 3.0;
                average_cost += price * sum_demand; // 累加成本
            }

            cout << "Charging cost under average scenario = " << average_cost << endl;
            delete[] avg_base_load;  // 釋放記憶體

            // 輸出最終充電排程與驗證
            cout << "\n=== Charging Schedule ===\n";
            for (int i = 0; i < nUsers; i++) {
                int startTime = -1;
                for (int t = 0; t < nTimeSlots; t++) {
                    if (Sigma[i][t].get(GRB_DoubleAttr_X) > 0.5) {
                        startTime = t;  // 取得開始時段
                        break;
                    }
                }
                cout << "User " << i << "'s charging schedule (required energy: "
                     << fixed << setprecision(2) << requiredEnergy[i]
                     << " kWh, duration: " << duration[i]
                     << ", start time: " << startTime << "):" << endl;
                bool hasCharging = false;
                double totalCharged = 0.0;
                for (int t = 0; t < nTimeSlots; t++) {
                    double power = chargingPower[i][t].get(GRB_DoubleAttr_X);
                    totalCharged += power;  // 累加實際充電量
                }
                cout << "Total charged energy: " << fixed << setprecision(2)
                     << totalCharged << " kWh (Required: " << requiredEnergy[i] << " kWh)" << endl;
                for (int t = startTime; t < startTime + duration[i] && t < nTimeSlots; t++) {
                    double power = chargingPower[i][t].get(GRB_DoubleAttr_X);
                    if (power > epsilon) { // 只顯示有效充電
                        cout << "Time slot " << t << ": "
                             << fixed << setprecision(2) << power << " kW" << endl;
                        hasCharging = true;
                    }
                }
                if (!hasCharging && startTime != -1)
                    cout << "No charging activity within active period" << endl;
                else if (startTime == -1)
                    cout << "No charging activity" << endl;
                if (abs(totalCharged - requiredEnergy[i]) > 1e-6)
                    cout << "WARNING: Energy constraint not satisfied for user " << i << endl;
                cout << endl;
            }
        } else {
            cout << "Optimization status: " << status << endl;  // 非最佳狀態
        }

        // 清理記憶體
        for (int s = 0; s < maxScenarios; s++) delete[] base_load_set[s];
        delete[] base_load_set;
        for (int i = 0; i < nUsers; i++) {
            delete[] chargingPower[i];
            delete[] batteryLevel[i];
            delete[] Sigma[i];
            delete[] I[i];
        }
        delete[] chargingPower;
        delete[] batteryLevel;
        delete[] Sigma;
        delete[] I;
        delete[] duration;
        delete[] requiredEnergy;
    }
    catch (GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl; // Gurobi 錯誤代碼
        cout << e.getMessage() << endl;                    // 錯誤訊息
    }
    catch (...) {
        cout << "Exception during optimization" << endl;   // 其他例外狀況
    }

    return 0;  // 程式結束
}

