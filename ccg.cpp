#include <gurobi_c++.h>      // �ޤJ Gurobi C++ API ���Y��
#include <vector>           // �ޤJ�V�q�e���䴩
#include <random>           // �ޤJ�H���Ʋ��;��䴩
#include <ctime>            // �ޤJ�ɶ��禡
#include <cstdlib>          // �ޤJ�зǨ禡�w
#include <fstream>          // �ޤJ�ɮ׿�J��X�䴩
#include <iostream>         // �ޤJ��X�J��y�䴩
#include <sstream>          // �ޤJ�r���y�䴩
#include <time.h>           // �ޤJ�ª��ɶ��禡
#include <algorithm>        // �ޤJ�t��k�䴩�]�p sort�^
#include <iomanip>          // �ޤJ�榡�ƿ�X�䴩
using namespace std;        // �ϥμзǩR�W�Ŷ�

// ���U�禡�G�]�w�ܼƩά����W�� (�t�ϥΪ̽s�� i �M�ɬq t)
string setVarOrConstrName(string str, int i, int t) {
    ostringstream vname;    // �إߦr���y
    vname << str << "[" << i << "]" << "[" << t << "]";  // �զX�W��
    return vname.str();     // �^�Ǧr��
}

// ���U�禡�G�]�w�ܼƩά����W�� (�t�ϥΪ̽s�� i)
string setVarOrConstrName(string str, int i) {
    ostringstream vname;    
    vname << str << "[" << i << "]";   // �զX�W��
    return vname.str();     
}

// ���U�禡�G�]�w�ܼƩά����W�� (�t�ϥΪ̽s�� i�Bj �M�ɬq t)
string setVarOrConstrName(string str, int i, int j, int t) {
    ostringstream vname;    
    vname << str << "[" << i << "]" << "[" << j << "]" << "[" << t << "]";  // �զX�W��
    return vname.str();     
}

int main(int argc, char *argv[])
{
    // �O���{���}�l�ɶ�
    clock_t t1, t2;          
    t1 = clock();            // Ū����l�ɶ�

    // ��l�Ʊ`��
    const double maxChargingPower = 50.0;              // �̤j�R�q�\�v (kW)
    const double bignumber = maxChargingPower;         // �j M ��k��
    const double maxBaseLoad = 220.0;                  // �̤j��� (kW)
    const int nTimeSlots = 24;                         // �ɬq��
    const double batteryCapacity = 60.4;               // �q���e�q (kWh)
    const double initialSOC = batteryCapacity / 2;     // ��l�q�q (�@�b�e�q)
    int worstScenIndex = 0;                            // ���a���ү���
    const int maxScenarios = 100;                      // �]�w���ҤW��
    const double epsilon = 1e-10;                      // ��ת��e

        // ���ͨϥΪ̩M���Ҽƶq
    int nUsers = 5;                                   // �ϥΪ̼�
    int nScenarios = 50;                              // ���Ҽ� 
    double maxLoad = maxBaseLoad + nUsers * maxChargingPower; // �̤j�`�t��
    int *duration = new int[nUsers];                  // �C��ϥΪ̥R�q�ɪ�
    double *requiredEnergy = new double[nUsers];      // �C��ϥΪ̻ݨD��q
    cout << "Users: " << nUsers << ", Scenarios: " << nScenarios << endl; // �C�L�Ѽ�

        // �ϥεw�s�X��������Ҹ�� (�d�� 80�V220)
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

   // === �ϥΪ̻ݨD��� (25�H�T�w) ===
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


    // �C�L���D�W��
    cout << "Total users: " << nUsers << endl;        
    cout << "Total time slots: " << nTimeSlots << endl;
    cout << "Total scenarios: " << nScenarios << endl;

    try {
        // �إ� Gurobi ���һP�ҫ�
        GRBEnv env = GRBEnv();                        
        GRBModel master_model = GRBModel(env);         // �إߥD���D�ҫ�
        master_model.set("NumericFocus", "3");         // ���ɼƭȺ��
        master_model.set(GRB_IntParam_Threads, 0);     // �۰ʳ]�w�������
        master_model.set(GRB_DoubleParam_MIPGap, 0.05);// �]�w���\ Gap
        master_model.set(GRB_DoubleParam_FeasibilityTol, 1e-9);// �i��ʤ��t

        // �w�q�M���ܼ�
        GRBVar **chargingPower = new GRBVar *[nUsers]; // �R�q�\�v
        GRBVar **batteryLevel = new GRBVar *[nUsers];  // �q���q�q
        GRBVar **Sigma = new GRBVar *[nUsers];         // �Ұʫ��ܾ�
        GRBVar **I = new GRBVar *[nUsers];             // �E�����ܾ�
        for (int i = 0; i < nUsers; i++) {z/
            chargingPower[i] = new GRBVar[nTimeSlots]; 
            batteryLevel[i] = new GRBVar[nTimeSlots + 1]; 
            Sigma[i] = new GRBVar[nTimeSlots];        
            I[i] = new GRBVar[nTimeSlots];           
            for (int t = 0; t < nTimeSlots; t++) {
                // �s���ܼơG�R�q�\�v
                chargingPower[i][t] = master_model.addVar(0.0, maxChargingPower, 0.0, GRB_CONTINUOUS,
                    setVarOrConstrName("ChargingPower", i, t));
                // �G���ܼơG�}�l�R�q����
                Sigma[i][t] = master_model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                    setVarOrConstrName("Sigma", i, 0, t));
                // �G���ܼơG���D���A����
                I[i][t] = master_model.addVar(0.0, 1.0, 0.0, GRB_BINARY,
                    setVarOrConstrName("I", i, 0, t));
            }
            for (int t = 0; t <= nTimeSlots; t++) {
                // �s���ܼơG�q���q�q
                batteryLevel[i][t] = master_model.addVar(0.0, batteryCapacity, 0.0, GRB_CONTINUOUS,
                    setVarOrConstrName("BatteryLevel", i, t));
            }
        }
        // �����ܼơG���a���Ҧ���
        GRBVar eta = master_model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "eta");
        master_model.update();  // ��s�ҫ�

        // �q���ʺA�P�`��q����
        for (int i = 0; i < nUsers; i++) {
            // ��l�q�q�]�w
            master_model.addConstr(batteryLevel[i][0] == initialSOC, setVarOrConstrName("InitialSOC", i));
            for (int t = 1; t <= nTimeSlots; t++) {
                // �q�q�H�R�q�\�v�ֿn�ܤ�
                master_model.addConstr(batteryLevel[i][t] == batteryLevel[i][t - 1] + chargingPower[i][t - 1],
                    setVarOrConstrName("BatteryUpdate", i, t));
            }
            // �`�R�q��q����ݨD
            GRBLinExpr totalEnergy = 0;
            for (int t = 0; t < nTimeSlots; t++) totalEnergy += chargingPower[i][t];
            master_model.addConstr(totalEnergy == requiredEnergy[i], setVarOrConstrName("TotalEnergy", i));
        }

        // �}�l�ɬq�u���@������
        for (int i = 0; i < nUsers; i++) {
            GRBLinExpr startConstr = 0;
            // �L�k�b�L�Įɬq�}�l
            for (int t = nTimeSlots - duration[i] + 1; t < nTimeSlots; t++) {
                master_model.addConstr(Sigma[i][t] == 0, setVarOrConstrName("StartConstr", i, 0, t));
            }
            // �����B�u���@���}�l�ɬq
            for (int t = 0; t <= nTimeSlots - duration[i]; t++) startConstr += Sigma[i][t];
            master_model.addConstr(startConstr == 1, setVarOrConstrName("StartOnce", i));
        }

        // ���D���ܾ�
        for (int i = 0; i < nUsers; i++) {
            for (int t = 0; t < nTimeSlots; t++) {
                GRBLinExpr activeConstr = 0;
                // �p�G�b���İ϶����Y�������D
                for (int start = max(0, t - duration[i] + 1); start <= t; start++)
                    activeConstr += Sigma[i][start];
                master_model.addConstr(I[i][t] == activeConstr, setVarOrConstrName("ActiveConstr", i, 0, t));
            }
        }
        master_model.update();  // ��s�ҫ�

        // C&CG �D�j����
        double Upper_bound = 99999999999.0;   
        double Lower_bound = 0.0;            
        double gap = 10;                     
        double best_Upper_bound = GRB_INFINITY;//�C�����N��s�᪺�W�ɭ� 
        vector<double> worst_base_load;     
        vector<vector<GRBVar>> Load;       
        int iter = 0;                       

        // �j�����G�W�U�ɮt�Z�j���H��
        while (best_Upper_bound - Lower_bound >= gap) {
            cout << "\nIteration " << iter << " Master start\n";  // �C�L���N�T��
            Load.resize(iter + 1);                     // �s�W��e���N�� Load �ܼƪŶ�
            Load.at(iter).resize(nTimeSlots);         
            for (int t = 0; t < nTimeSlots; t++) {
                // �w�q�C�Ӯɬq�� Load �ܼ�
                Load.at(iter).at(t) = master_model.addVar(0.0, maxLoad, 0.0, GRB_CONTINUOUS,
                    setVarOrConstrName("Load", iter, t));
            }
            master_model.update();  // ��s�ҫ�

            // �`�t������ (�[�W���)
            for (int t = 0; t < nTimeSlots; t++) {
                GRBLinExpr TotalLoad_RHS = 0;
                for (int i = 0; i < nUsers; i++) TotalLoad_RHS += chargingPower[i][t];
                // �Ĥ@���βĤ@�Ӱ�����ҡA����γ��a���
                if (iter == 0) TotalLoad_RHS += base_load_set[0][t];
                else TotalLoad_RHS += worst_base_load[t];
                master_model.addConstr(Load.at(iter).at(t) == TotalLoad_RHS,
                    setVarOrConstrName("Master_Load_Constr", iter, t));
            }
            master_model.update();  // ��s�ҫ�

            // �ؼШ�ƪ��ǳơG�G��������
            GRBQuadExpr obj_constr = 0;
            for (int t = 0; t < nTimeSlots; t++) {
                GRBLinExpr price = 0.02 * Load.at(iter).at(t) + 3;  // ������
                for (int i = 0; i < nUsers; i++) obj_constr += price * chargingPower[i][t];
            }
            master_model.addQConstr(eta >= obj_constr, setVarOrConstrName("Master_obj", iter));
            master_model.update();  // ��s�ҫ�

            // �]�w�èD�ѥD���D
            master_model.setObjective(eta, GRB_MINIMIZE);  
            master_model.write(to_string(iter) + "master.lp"); // ��X LP ��
            master_model.optimize();  // �̤p��
            Lower_bound = master_model.get(GRB_DoubleAttr_ObjVal); // ��s�U��

            // �l���D�G�w��C�Ӱ�����ҭp�⦨��
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
                if (cost > max_cost) {  // ��X�̤j����
                    max_cost = cost;
                    worstScenIndex = s;
                }
            }
            Upper_bound = max_cost;  
            best_Upper_bound = min(best_Upper_bound, max_cost); // ��s�W��

            // ��s���a�������
            worst_base_load.clear();
            for (int t = 0; t < nTimeSlots; t++)
                worst_base_load.push_back(base_load_set[worstScenIndex][t]);

            cout << "iter=" << iter << ", max_cost: " << max_cost
                 << ", best_Upper_bound: " << best_Upper_bound
                 << ", Lower_bound: " << Lower_bound << endl;
            iter++;  // �i�J�U�@���N
        }

        // �O�������ɶ��ÿ�X
        t2 = clock();
        printf("Total time: %lf\n", (t2 - t1) / (double)(CLOCKS_PER_SEC)); // ��X�`�Ӯ�

        // �ˬd�D�Ѫ��A
        int status = master_model.get(GRB_IntAttr_Status);
        if (status == GRB_OPTIMAL) {
            double ObjValue = master_model.get(GRB_DoubleAttr_ObjVal); // �̨ΥؼЭ�
            cout << "Total charging cost = " << ObjValue << endl;

            // �������ҤU���R�q�����p��
            double* avg_base_load = new double[nTimeSlots];
            for (int t = 0; t < nTimeSlots; t++) {
                avg_base_load[t] = 0.0;
                for (int s = 0; s < nScenarios; s++)
                    avg_base_load[t] += base_load_set[s][t];
                avg_base_load[t] /= nScenarios; // �p�⥭�����
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
                average_cost += price * sum_demand; // �֥[����
            }

            cout << "Charging cost under average scenario = " << average_cost << endl;
            delete[] avg_base_load;  // ����O����

            // ��X�̲ץR�q�Ƶ{�P����
            cout << "\n=== Charging Schedule ===\n";
            for (int i = 0; i < nUsers; i++) {
                int startTime = -1;
                for (int t = 0; t < nTimeSlots; t++) {
                    if (Sigma[i][t].get(GRB_DoubleAttr_X) > 0.5) {
                        startTime = t;  // ���o�}�l�ɬq
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
                    totalCharged += power;  // �֥[��ڥR�q�q
                }
                cout << "Total charged energy: " << fixed << setprecision(2)
                     << totalCharged << " kWh (Required: " << requiredEnergy[i] << " kWh)" << endl;
                for (int t = startTime; t < startTime + duration[i] && t < nTimeSlots; t++) {
                    double power = chargingPower[i][t].get(GRB_DoubleAttr_X);
                    if (power > epsilon) { // �u��ܦ��ĥR�q
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
            cout << "Optimization status: " << status << endl;  // �D�̨Ϊ��A
        }

        // �M�z�O����
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
        cout << "Error code = " << e.getErrorCode() << endl; // Gurobi ���~�N�X
        cout << e.getMessage() << endl;                    // ���~�T��
    }
    catch (...) {
        cout << "Exception during optimization" << endl;   // ��L�ҥ~���p
    }

    return 0;  // �{������
}

