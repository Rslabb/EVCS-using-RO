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
string setVarOrConstrName(string str, int i, int t) {  // 定義幫助函式，用來產生變數或約束的名稱，包含使用者 i 和時段 t
    ostringstream vname;    // 建立字串串流
    vname << str << "[" << i << "]" << "[" << t << "]";  // 組合名稱
    return vname.str();     // 回傳字串
}

// 幫助函式：設定變數或約束名稱 (含使用者編號 i)
string setVarOrConstrName(string str, int i) {  // 定義幫助函式，用來產生變數或約束的名稱，只包含使用者 i
    ostringstream vname;    
    vname << str << "[" << i << "]";   // 組合名稱
    return vname.str();     
}

// 幫助函式：設定變數或約束名稱 (含使用者編號 i、j 和時段 t)
string setVarOrConstrName(string str, int i, int j, int t) {  // 定義幫助函式，用來產生變數或約束的名稱，包含使用者 i、j 和時段 t
    ostringstream vname;    
    vname << str << "[" << i << "]" << "[" << j << "]" << "[" << t << "]";  // 組合名稱
    return vname.str();     
}

int main(int argc, char *argv[])  // 主程式入口，接收命令列參數
{
    // 記錄程式開始時間
    clock_t t1, t2;          // 宣告兩個時間變數，用來記錄開始和結束時間
    t1 = clock();            // 讀取初始時間

    // 初始化常數
    const double maxChargingPower = 50.0;              // 最大充電功率 (kW)
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

        // 基載情境資料
    double base_load[50][24] ={  // 定義基載情境的二維陣列，包含50個情境，每個情境有24個時段的基載值
            {200,195,190,185,180,185,195,200,190,180,165,158,150,142,137,132,128,122,112,106,108,110,108,106},  // 第一個情境的基載資料
            {198,193,188,183,178,183,193,198,188,178,163,156,148,140,135,130,126,120,110,106,108,110,108,106},  // 第二個情境的基載資料
            {196,191,186,181,176,181,191,196,186,176,161,154,146,138,133,128,124,118,108,106,108,110,108,106},  // 第三個情境的基載資料
            {194,189,184,179,174,179,189,194,184,174,159,152,144,136,131,126,122,116,106,106,108,110,108,106},  // 第四個情境的基載資料
            {192,187,182,177,172,177,187,192,182,172,157,150,142,134,129,124,120,114,104,106,108,110,108,106},  // 第五個情境的基載資料
            {190,185,180,175,170,175,185,190,180,170,155,148,140,132,127,122,118,112,102,106,108,110,108,106},  // 第六個情境的基載資料
            {188,183,178,173,168,173,183,188,178,168,153,146,138,130,125,120,116,110,100,106,108,110,108,106},  // 第七個情境的基載資料
            {186,181,176,171,166,171,181,186,176,166,151,144,136,128,123,118,114,108,98,106,108,110,108,106},  // 第八個情境的基載資料
            {184,179,174,169,164,169,179,184,174,164,149,142,134,126,121,116,112,106,96,106,108,110,108,106},  // 第九個情境的基載資料
            {182,177,172,167,162,167,177,182,172,162,147,140,132,124,119,114,110,104,94,106,108,110,108,106},  // 第十個情境的基載資料
            {180,175,170,165,160,165,175,180,170,160,145,138,130,122,117,112,108,102,92,106,108,110,108,106},  // 第十一個情境的基載資料
            {178,173,168,163,158,163,173,178,168,158,143,136,128,120,115,110,106,100,90,106,108,110,108,106},  // 第十二個情境的基載資料
            {176,171,166,161,156,161,171,176,166,156,141,134,126,118,113,108,104,98,88,106,108,110,108,106},  // 第十三個情境的基載資料
            {174,169,164,159,154,159,169,174,164,154,139,132,124,116,111,106,102,96,86,106,108,110,108,106},  // 第十四個情境的基載資料
            {172,167,162,157,152,157,167,172,162,152,137,130,122,114,109,104,100,94,84,106,108,110,108,106},  // 第十五個情境的基載資料
            {170,165,160,155,150,155,165,170,160,150,135,128,120,112,107,102,98,92,82,106,108,110,108,106},  // 第十六個情境的基載資料
            {168,163,158,153,148,153,163,168,158,148,133,126,118,110,105,100,96,90,80,106,108,110,108,106},  // 第十七個情境的基載資料
            {166,161,156,151,146,151,161,166,156,146,131,124,116,108,103,98,94,88,80,106,108,110,108,106},  // 第十八個情境的基載資料
            {164,159,154,149,144,149,159,164,154,144,129,122,114,106,101,96,92,86,80,106,108,110,108,106},  // 第十九個情境的基載資料
            {162,157,152,147,142,147,157,162,152,142,127,120,112,104,99,94,90,84,80,106,108,110,108,106},  // 第二十個情境的基載資料
            {160,155,150,145,140,145,155,160,150,140,125,118,110,102,97,92,88,82,80,106,108,110,108,106},  // 第二十一個情境的基載資料
            {158,153,148,143,138,143,153,158,148,138,123,116,108,100,95,90,86,80,80,106,108,110,108,106},  // 第二十二個情境的基載資料
            {156,151,146,141,136,141,151,156,146,136,121,114,106,98,93,88,84,80,80,106,108,110,108,106},  // 第二十三個情境的基載資料
            {154,149,144,139,134,139,149,154,144,134,119,112,104,96,91,86,82,80,80,106,108,110,108,106},  // 第二十四個情境的基載資料
            {152,147,142,137,132,137,147,152,142,132,117,110,102,94,89,84,80,80,80,106,108,110,108,106},  // 第二十五個情境的基載資料
            {150,145,140,135,130,135,145,150,140,130,115,108,100,92,87,82,80,80,80,106,108,110,108,106},  // 第二十六個情境的基載資料
            {148,143,138,133,128,133,143,148,138,128,113,106,98,90,85,80,80,80,80,106,108,110,108,106},  // 第二十七個情境的基載資料
            {146,141,136,131,126,131,141,146,136,126,111,104,96,88,83,80,80,80,80,106,108,110,108,106},  // 第二十八個情境的基載資料
            {144,139,134,129,124,129,139,144,134,124,109,102,94,86,81,80,80,80,80,106,108,110,108,106},  // 第二十九個情境的基載資料
            {142,137,132,127,122,127,137,142,132,122,107,100,92,84,79,80,80,80,80,106,108,110,108,106},  // 第三十個情境的基載資料
            {140,135,130,125,120,125,135,140,130,120,105,98,90,82,77,80,80,80,80,106,108,110,108,106},  // 第三十一個情境的基載資料
            {138,133,128,123,118,123,133,138,128,118,103,96,88,80,75,80,80,80,80,106,108,110,108,106},  // 第三十二個情境的基載資料
            {136,131,126,121,116,121,131,136,126,116,101,94,86,78,73,80,80,80,80,106,108,110,108,106},  // 第三十三個情境的基載資料
            {134,129,124,119,114,119,129,134,124,114,99,92,84,76,71,80,80,80,80,106,108,110,108,106},  // 第三十四個情境的基載資料
            {132,127,122,117,112,117,127,132,122,112,97,90,82,74,69,80,80,80,80,106,108,110,108,106},  // 第三十五個情境的基載資料
            {130,125,120,115,110,115,125,130,120,110,95,88,80,72,67,80,80,80,80,106,108,110,108,106},  // 第三十六個情境的基載資料
            {128,123,118,113,108,113,123,128,118,108,93,86,78,70,65,80,80,80,80,106,108,110,108,106},  // 第三十七個情境的基載資料
            {126,121,116,111,106,111,121,126,116,106,91,84,76,68,63,80,80,80,80,106,108,110,108,106},  // 第三十八個情境的基載資料
            {124,119,114,109,104,109,119,124,114,104,89,82,74,66,61,80,80,80,80,106,108,110,108,106},  // 第三十九個情境的基載資料
            {122,117,112,107,102,107,117,122,112,102,87,80,72,64,59,80,80,80,80,106,108,110,108,106},  // 第四十個情境的基載資料
            {120,115,110,105,100,105,115,120,110,100,85,78,70,62,57,80,80,80,80,106,108,110,108,106},  // 第四十一個情境的基載資料
            {118,113,108,103,98,103,113,118,108,98,83,76,68,60,55,80,80,80,80,106,108,110,108,106},  // 第四十二個情境的基載資料
            {116,111,106,101,96,101,111,116,106,96,81,74,66,58,53,80,80,80,80,106,108,110,108,106},  // 第四十三個情境的基載資料
            {114,109,104,99,94,99,109,114,104,94,79,72,64,56,51,80,80,80,80,106,108,110,108,106},  // 第四十四個情境的基載資料
            {112,107,102,97,92,97,107,112,102,92,77,70,62,54,49,80,80,80,80,106,108,110,108,106},  // 第四十五個情境的基載資料
            {110,105,100,95,90,95,105,110,100,90,75,68,60,52,47,80,80,80,80,106,108,110,108,106}   // 第四十六個情境的基載資料
        };  // 基載陣列結束


    double **base_load_set = new double *[maxScenarios];  // 動態分配基載情境的二維陣列指標
    for (int s = 0; s < maxScenarios; s++) {  // 迴圈遍歷所有最大情境數
        base_load_set[s] = new double[nTimeSlots];  // 為每個情境分配24個時段的陣列
        for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
            base_load_set[s][t] = base_load[s][t];  // 複製基載資料到新陣列
        }
    }

   // === 使用者需求資料 ===
    int hard_duration[25] = {  // 定義使用者充電時長陣列，包含25個值
        4,6,3,5,7,2,8,6,4,3,  // 前10個使用者的充電時長
        5,7,2,6,4,8,3,5,7,6,  // 接下來10個使用者的充電時長
        4,2,5,3,7  // 最後5個使用者的充電時長
    };

    double hard_requiredEnergy[25] = {  // 定義使用者需求能量陣列，包含25個值
        12.5,14.0,11.2,13.8,10.5,12.0,15.0,13.2,11.8,12.6,  // 前10個使用者的需求能量
        14.4,13.7,10.9,12.3,11.5,14.9,13.1,12.8,10.7,13.4,  // 接下來10個使用者的需求能量
        11.9,12.2,13.6,14.1,10.8  // 最後5個使用者的需求能量
    };

    for (int i = 0; i < nUsers; i++) {  // 迴圈遍歷每個使用者
        duration[i] = hard_duration[i];  // 複製充電時長到動態陣列
        requiredEnergy[i] = hard_requiredEnergy[i];  // 複製需求能量到動態陣列
        if (initialSOC + requiredEnergy[i] > batteryCapacity) {  // 檢查是否超過電池容量
            requiredEnergy[i] = batteryCapacity - initialSOC - 0.1;  // 調整需求能量，避免超過容量
        }
    }

    cout << "Total users: " << nUsers << endl;        // 輸出總使用者數
    cout << "Total time slots: " << nTimeSlots << endl;  // 輸出總時段數
    cout << "Total scenarios: " << nScenarios << endl;  // 輸出總情境數

    try {  // 開始 try 區塊，捕捉 Gurobi 例外
        // 建立 Gurobi 環境與模型
        GRBEnv env = GRBEnv();                        // 建立 Gurobi 環境
        GRBModel master_model = GRBModel(env);         // 建立主問題模型
        master_model.set("NumericFocus", "3");         // 提升數值精度
        master_model.set(GRB_IntParam_Threads, 0);     // 自動設定執行緒數
        master_model.set(GRB_DoubleParam_MIPGap, 0.05);// 設定允許 Gap
        master_model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);// 可行性公差

        // 定義決策變數
        GRBVar **chargingPower = new GRBVar *[nUsers]; // 充電功率
        GRBVar **batteryLevel = new GRBVar *[nUsers];  // 電池電量
        GRBVar **Sigma = new GRBVar *[nUsers];         // 開始充電指示 
        GRBVar **I = new GRBVar *[nUsers];             // 是否正在充電指示器
        for (int i = 0; i < nUsers; i++) {  // 迴圈遍歷每個使用者
            chargingPower[i] = new GRBVar[nTimeSlots];   // 分配充電功率陣列
            batteryLevel[i] = new GRBVar[nTimeSlots + 1];  // 分配電池電量陣列，多一個初始值
            Sigma[i] = new GRBVar[nTimeSlots];         // 分配開始充電指示陣列
            I[i] = new GRBVar[nTimeSlots];             // 分配充電中指示陣列
            for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                // 連續變數：充電功率
                chargingPower[i][t] = master_model.addVar(0.0, maxChargingPower, 0.0, GRB_CONTINUOUS,  // 新增充電功率變數，為連續變數，上限為最大充電功率
                    setVarOrConstrName("ChargingPower", i, t));  // 設定變數名稱
                // 二元變數：開始充電指示
                Sigma[i][t] = master_model.addVar(0.0, 1.0, 0.0, GRB_BINARY,  // 新增開始充電指示變數，為二元變數
                    setVarOrConstrName("Sigma", i, 0, t));  // 設定變數名稱 
                // 二元變數：是否正在充電指示
                I[i][t] = master_model.addVar(0.0, 1.0, 0.0, GRB_BINARY,  // 新增充電中指示變數，為二元變數
                    setVarOrConstrName("I", i, 0, t));  // 設定變數名稱 
            }
            for (int t = 0; t <= nTimeSlots; t++) {  // 迴圈遍歷每個時段，包括初始
                // 連續變數：電池電量
                batteryLevel[i][t] = master_model.addVar(0.0, batteryCapacity, 0.0, GRB_CONTINUOUS,  // 新增電池電量變數，為連續變數，上限為電池容量
                    setVarOrConstrName("BatteryLevel", i, t));  // 設定變數名稱
            }
        }
        // 極值變數：最壞情境成本
        GRBVar eta = master_model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "eta");  // 新增 eta 變數，代表最壞情境的成本，為連續變數
        master_model.update();  // 更新模型，讓新變數生效

        // 電池動態與總能量約束
        for (int i = 0; i < nUsers; i++) {  // 迴圈遍歷每個使用者
            // 初始電量設定
            master_model.addConstr(batteryLevel[i][0] == initialSOC, setVarOrConstrName("InitialSOC", i));  // 新增約束：初始電池電量等於初始 SOC
            for (int t = 1; t <= nTimeSlots; t++) {  // 迴圈遍歷每個時段，從1開始
                // 電量隨充電功率累積變化
                master_model.addConstr(batteryLevel[i][t] == batteryLevel[i][t - 1] + chargingPower[i][t - 1],  // 新增約束：電池電量更新為前一時段加上充電功率
                    setVarOrConstrName("BatteryUpdate", i, t));  // 設定約束名稱
            }
            // 總充電能量等於需求
            GRBLinExpr totalEnergy = 0;  // 建立線性表達式，用來累加總能量
            for (int t = 0; t < nTimeSlots; t++) totalEnergy += chargingPower[i][t];  // 累加每個時段的充電功率
            master_model.addConstr(totalEnergy == requiredEnergy[i], setVarOrConstrName("TotalEnergy", i));  // 新增約束：總能量等於需求能量
        }

        // 開始時段只能選一次約束
        for (int i = 0; i < nUsers; i++) {  // 迴圈遍歷每個使用者
            GRBLinExpr startConstr = 0;  // 建立線性表達式，用來累加開始指示
            // 無法在無效時段開始
            for (int t = nTimeSlots - duration[i] + 1; t < nTimeSlots; t++) {  // 迴圈遍歷無效開始時段
                master_model.addConstr(Sigma[i][t] == 0, setVarOrConstrName("StartConstr", i, 0, t));  // 新增約束：無效時段不能開始充電
            }
            // 必須且只能選一次開始時段
            for (int t = 0; t <= nTimeSlots - duration[i]; t++) startConstr += Sigma[i][t];  // 累加有效開始時段的指示
            master_model.addConstr(startConstr == 1, setVarOrConstrName("StartOnce", i));  // 新增約束：開始指示總和等於1
        }

        // 正在充電指示器
        for (int i = 0; i < nUsers; i++) {  // 迴圈遍歷每個使用者
            for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                GRBLinExpr activeConstr = 0;  // 建立線性表達式，用來累加活躍開始指示
                // 如果在有效區間內
                for (int start = max(0, t - duration[i] + 1); start <= t; start++)  // 迴圈遍歷可能的開始時段
                    activeConstr += Sigma[i][start];  // 累加開始指示
                master_model.addConstr(I[i][t] == activeConstr, setVarOrConstrName("ActiveConstr", i, 0, t));  // 新增約束：充電中指示等於累加值
            }
        }
        master_model.update();  // 更新模型，讓新約束生效

        // C&CG 主迴圈初值
        double Upper_bound = 99999999999.0;   // 設定初始上界為很大值
        double Lower_bound = 0.0;            // 設定初始下界為0
        double gap = 10;                     // 設定收斂差距門檻
        double best_Upper_bound = GRB_INFINITY;//每次迭代更新後的上界值 
        vector<double> worst_base_load;     // 儲存最壞基載情境的向量
        vector<vector<GRBVar>> Load;       // 儲存每個迭代的 Load 變數的二維向量
        int iter = 0;                       // 迭代計數器初始化為0

        // 迴圈條件：上下界差距大於閾值
        while (best_Upper_bound - Lower_bound >= gap) {  // 當最佳上界減下界大於差距時繼續迴圈
            cout << "\nIteration " << iter << " Master start\n";  // 列印迭代訊息
            Load.resize(iter + 1);                     // 新增當前迭代的 Load 變數空間
            Load.at(iter).resize(nTimeSlots);         // 為當前迭代分配時段數的變數空間
            for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                // 定義每個時段的 Load 變數
                Load.at(iter).at(t) = master_model.addVar(0.0, maxLoad, 0.0, GRB_CONTINUOUS,  // 新增 Load 變數，為連續變數，上限為最大負載
                    setVarOrConstrName("Load", iter, t));  // 設定變數名稱
            }
            master_model.update();  // 更新模型，讓新變數生效

            // 總負載約束 (加上基載)
            for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                GRBLinExpr TotalLoad_RHS = 0;  // 建立線性表達式，用來計算總負載右側
                for (int i = 0; i < nUsers; i++) TotalLoad_RHS += chargingPower[i][t];  // 累加所有使用者的充電功率
                // 第一次用第一個基載情境，之後用最壞基載
                if (iter == 0) TotalLoad_RHS += base_load_set[0][t];  // 如果是第一次迭代，使用第一個情境的基載
                else TotalLoad_RHS += worst_base_load[t];  // 否則使用最壞基載
                master_model.addConstr(Load.at(iter).at(t) == TotalLoad_RHS,  // 新增約束： Load 等於總負載
                    setVarOrConstrName("Master_Load_Constr", iter, t));  // 設定約束名稱
            }
            master_model.update();  // 更新模型，讓新約束生效

            // 目標函數的準備：二次式約束
            GRBQuadExpr obj_constr = 0;  // 建立二次表達式，用來計算目標
            for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                GRBLinExpr price = 0.02 * Load.at(iter).at(t) + 3;  // 計算價格：0.02 * Load + 3
                for (int i = 0; i < nUsers; i++) obj_constr += price * chargingPower[i][t];  // 累加價格乘以每個使用者的充電功率
            }
            master_model.addQConstr(eta >= obj_constr, setVarOrConstrName("Master_obj", iter));  // 新增二次約束： eta 大於等於 obj_constr
            master_model.update();  // 更新模型，讓新約束生效

            // 設定並求解主問題
            master_model.setObjective(eta, GRB_MINIMIZE);   // 設定目標函數為最小化 eta
            master_model.write(to_string(iter) + "master.lp"); // 輸出 LP 檔，用來儲存模型
            master_model.optimize();  // 求解模型
            Lower_bound = master_model.get(GRB_DoubleAttr_ObjVal); // 更新下界為目標值

            // 子問題：針對每個基載情境計算成本
            double max_cost = 0;  // 初始化最大成本為0
            worstScenIndex = 0;  // 初始化最壞情境索引為0
            for (int s = 0; s < nScenarios; s++) {  // 迴圈遍歷每個情境
                double cost = 0;  // 初始化當前情境成本為0
                for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                    double total_load = base_load_set[s][t];  // 取得當前情境和時段的基載
                    for (int i = 0; i < nUsers; i++)  // 迴圈遍歷每個使用者
                        total_load += chargingPower[i][t].get(GRB_DoubleAttr_X);  // 累加充電功率到總負載
                    double price = 0.02 * total_load + 2;  // 計算價格：0.02 * total_load + 2 (注意：這裡+2，與主問題+3不同，可能為模型設計)
                    for (int i = 0; i < nUsers; i++)  // 迴圈遍歷每個使用者
                        cost += price * chargingPower[i][t].get(GRB_DoubleAttr_X);  // 累加價格乘以充電功率到成本
                }
                if (cost > max_cost) {  // 如果當前成本大於最大成本
                    max_cost = cost;  // 更新最大成本
                    worstScenIndex = s;  // 更新最壞情境索引
                }
            }
            Upper_bound = max_cost;   // 更新上界為最大成本
            best_Upper_bound = min(best_Upper_bound, max_cost); // 更新最佳上界為最小值

            // 更新最壞基載情境
            worst_base_load.clear();  // 清空最壞基載向量
            for (int t = 0; t < nTimeSlots; t++)  // 迴圈遍歷每個時段
                worst_base_load.push_back(base_load_set[worstScenIndex][t]);  // 加入最壞情境的基載值

            cout << "iter=" << iter << ", max_cost: " << max_cost  // 輸出迭代資訊
                 << ", best_Upper_bound: " << best_Upper_bound
                 << ", Lower_bound: " << Lower_bound << endl;
            iter++;  // 迭代計數器加1
        }

        // 記錄結束時間並輸出
        t2 = clock();  // 讀取結束時間
        printf("Total time: %lf\n", (t2 - t1) / (double)(CLOCKS_PER_SEC)); // 輸出總耗時

        // 檢查求解狀態
        int status = master_model.get(GRB_IntAttr_Status);  // 取得優化狀態
        if (status == GRB_OPTIMAL) {  // 如果是最佳解
            double ObjValue = master_model.get(GRB_DoubleAttr_ObjVal); // 取得最佳目標值
            cout << "Total charging cost = " << ObjValue << endl;  // 輸出總充電成本

            // 平均情境下的充電成本計算
            double* avg_base_load = new double[nTimeSlots];  // 分配平均基載陣列
            for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                avg_base_load[t] = 0.0;  // 初始化平均值為0
                for (int s = 0; s < nScenarios; s++)  // 迴圈遍歷每個情境
                    avg_base_load[t] += base_load_set[s][t];  // 累加所有情境的基載
                avg_base_load[t] /= nScenarios; // 計算平均基載
            }

            double average_cost = 0.0;  // 初始化平均成本為0
            for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                double total_load = avg_base_load[t];  // 取得平均基載
                double sum_demand = 0.0;  // 初始化需求總和為0
                for (int i = 0; i < nUsers; i++) {  // 迴圈遍歷每個使用者
                    double demand = chargingPower[i][t].get(GRB_DoubleAttr_X);  // 取得充電功率
                    total_load += demand;  // 累加到總負載
                    sum_demand += demand;  // 累加到需求總和
                }
                double price = 0.02 * total_load + 3.0;  // 計算價格：0.02 * total_load + 3
                average_cost += price * sum_demand; // 累加成本
            }

            cout << "Charging cost under average scenario = " << average_cost << endl;  // 輸出平均情境下的充電成本
            delete[] avg_base_load;  // 釋放平均基載陣列記憶體

            // 輸出最終充電排程與驗證
            cout << "\n=== Charging Schedule ===\n";  // 輸出充電排程標題
            for (int i = 0; i < nUsers; i++) {  // 迴圈遍歷每個使用者
                int startTime = -1;  // 初始化開始時間為-1
                for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                    if (Sigma[i][t].get(GRB_DoubleAttr_X) > 0.5) {  // 如果開始指示大於0.5
                        startTime = t;  // 設定開始時間
                        break;  // 跳出迴圈
                    }
                }
                cout << "User " << i << "'s charging schedule (required energy: "  // 輸出使用者資訊
                     << fixed << setprecision(2) << requiredEnergy[i]
                     << " kWh, duration: " << duration[i]
                     << ", start time: " << startTime << "):" << endl;
                bool hasCharging = false;  // 初始化是否有充電為 false
                double totalCharged = 0.0;  // 初始化總充電量為0
                for (int t = 0; t < nTimeSlots; t++) {  // 迴圈遍歷每個時段
                    double power = chargingPower[i][t].get(GRB_DoubleAttr_X);  // 取得充電功率
                    totalCharged += power;  // 累加總充電量
                }
                cout << "Total charged energy: " << fixed << setprecision(2)  // 輸出總充電量
                     << totalCharged << " kWh (Required: " << requiredEnergy[i] << " kWh)" << endl;
                for (int t = startTime; t < startTime + duration[i] && t < nTimeSlots; t++) {  // 迴圈遍歷充電期間
                    double power = chargingPower[i][t].get(GRB_DoubleAttr_X);  // 取得充電功率
                    if (power > epsilon) { // 只顯示有效充電
                        cout << "Time slot " << t << ": "  // 輸出時段充電功率
                             << fixed << setprecision(2) << power << " kW" << endl;
                        hasCharging = true;  // 設定有充電
                    }
                }
                if (!hasCharging && startTime != -1)  // 如果沒有充電但有開始時間
                    cout << "No charging activity within active period" << endl;  // 輸出無充電活動
                else if (startTime == -1)  // 如果沒有開始時間
                    cout << "No charging activity" << endl;  // 輸出無充電活動
                if (abs(totalCharged - requiredEnergy[i]) > 1e-6)  // 如果總充電量與需求不符
                    cout << "WARNING: Energy constraint not satisfied for user " << i << endl;  // 輸出警告
                cout << endl;  // 換行
            }
        } else {  // 如果不是最佳解
            cout << "Optimization status: " << status << endl;  // 輸出優化狀態
        }

        // 清理記憶體
        for (int s = 0; s < maxScenarios; s++) delete[] base_load_set[s];  // 釋放每個情境的基載陣列
        delete[] base_load_set;  // 釋放基載情境指標陣列
        for (int i = 0; i < nUsers; i++) {  // 迴圈遍歷每個使用者
            delete[] chargingPower[i];  // 釋放充電功率陣列
            delete[] batteryLevel[i];  // 釋放電池電量陣列
            delete[] Sigma[i];  // 釋放開始指示陣列
            delete[] I[i];  // 釋放充電中指示陣列
        }
        delete[] chargingPower;  // 釋放充電功率指標陣列
        delete[] batteryLevel;  // 釋放電池電量指標陣列
        delete[] Sigma;  // 釋放開始指示指標陣列
        delete[] I;  // 釋放充電中指示指標陣列
        delete[] duration;  // 釋放充電時長陣列
        delete[] requiredEnergy;  // 釋放需求能量陣列
    }
    catch (GRBException e) {  // 捕捉 Gurobi 例外
        cout << "Error code = " << e.getErrorCode() << endl; // Gurobi 錯誤代碼
        cout << e.getMessage() << endl;                    // 錯誤訊息
    }
    catch (...) {  // 捕捉其他例外
        cout << "Exception during optimization" << endl;   // 其他例外狀況
    }

    return 0;  // 程式結束，返回0表示成功
}



