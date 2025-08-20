# EVCS-using-RO
# ⚡ 基於智慧電網的充電排程管理 (PPO + 賽局理論 + C&CG)

這個專案主要研究 **電動車充電排程最佳化問題**，透過三種方法進行比較：  
- **強化學習 (PPO, Proximal Policy Optimization)**  
- **賽局理論 (Game Theory, Nash Equilibrium)**  
- **列與約束生成法 (C&CG, Column and Constraint Generation)**  

目標是 **降低充電成本、避免尖峰負載、並確保所有使用者需求被滿足**。

---

## 📌 功能
- 🤖 **強化學習 (PPO)**  
  - 使用策略梯度演算法  

- 🎲 **賽局理論模型**  
  - 透過納許均衡確保系統穩定  
  - 每位使用者獨立決策，確保需求滿足  

- 📐 **C&CG 演算法**  
  - 數學規劃方法  
  - 能保證找到全域最佳解  
  - 缺點是計算複雜度高，隨著規模增加需要更多時間  

---
Author:SU YU CHUN
Email: R76101112@gs.ncku.edu.tw
2025/8/20
## 📂 專案結構
```bash
├── src/                    
│   ├── ppo_model.py        # PPO 演算法實作
│   ├── game_theory.py      # 賽局理論模型
│   ├── ccg_solver.c++       # C&CG 演算法
├── requirements.txt        # 套件需求
└── README.md               # 說明文件


