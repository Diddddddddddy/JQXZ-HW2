"""
主程序入口模块
负责编排数据加载、模型训练、优化求解与结果评估的全流程。
新增功能：自动将实验结果保存至 run 文件夹，并附带时间戳。
"""

import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.config import Config
from utils.data_loader import DataProcessor
from models.predictor import DemandPredictor
from optimization.milp_solver import IRPSolver

class ExperimentRunner:
    """
    库存路径问题 (IRP) 实验控制器
    集成 Predict-then-Optimize (PTO) 范式的训练、评估与结果记录逻辑。
    """
    def __init__(self):
        # 1. 初始化配置与环境
        self.cfg = Config()
        torch.manual_seed(self.cfg.RANDOM_SEED)
        
        print(f"[Init] Initializing experiment environment...")
        
        # 2. 数据准备与拓扑生成
        self.data_processor = DataProcessor(self.cfg)
        self.dist_matrix = self.data_processor.generate_topology()
        self.train_loader, self.test_loader, self.input_dim = self.data_processor.load_data()
        
        # 3. 模型初始化
        self.model = DemandPredictor(
            input_dim=self.input_dim, 
            output_dim=self.cfg.NUM_STORES,
            hidden_dim=self.cfg.HIDDEN_DIM
        ).to(self.cfg.DEVICE)
        
        # 4. 优化器与损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # 5. 运筹优化求解器
        try:
            self.solver = IRPSolver(self.cfg, self.dist_matrix)
        except RuntimeError as e:
            print(
                "[Error] Failed to initialize MILP solver. "
                "This project requires Gurobi + gurobipy for the optimization stage.\n"
                f"[Error] Details: {e}\n"
                "[Hint] If you don't have Gurobi available, you can still review the code structure "
                "and model training part, but evaluation (MILP) cannot run."
            )
            raise SystemExit(1)

        # 6. 初始化结果保存目录
        self.run_dir = os.path.join(self.cfg.PROJECT_ROOT, "run")
        os.makedirs(self.run_dir, exist_ok=True)

    def train_pto(self):
        """
        执行 PTO 范式训练 (基于 MSE 损失)
        """
        print(f"\n{'='*20} Start PTO Training (MSE Loss) {'='*20}")
        self.model.train()
        
        for epoch in range(self.cfg.EPOCHS):
            epoch_loss = 0.0
            start_time = time.time()
            
            for X_batch, Y_batch in self.train_loader:
                X_batch = X_batch.to(self.cfg.DEVICE)
                Y_batch = Y_batch.to(self.cfg.DEVICE)
                
                # 前向传播
                preds = self.model(X_batch)
                loss = self.criterion(preds, Y_batch)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1:02d}/{self.cfg.EPOCHS}] "
                      f"MSE Loss: {avg_loss:.4f} | "
                      f"Time: {time.time()-start_time:.2f}s")
        
        print("Training Completed.")

    def evaluate(self, num_samples: int = 20):
        """
        全流程评估与结果记录
        
        Args:
            num_samples: 评估样本数量 (默认20以节省时间)
        """
        print(f"\n{'='*20} Start Evaluation (Top {num_samples} Samples) {'='*20}")
        self.model.eval()
        
        metrics = {
            "mse": [],
            "true_cost": [],
            "oracle_cost": [],
            "regret": []
        }
        
        print("Solving MILP instances... (Please wait)")
        with torch.no_grad():
            for i, (X, Y_true) in enumerate(tqdm(self.test_loader, total=num_samples)):
                if i >= num_samples:
                    break
                    
                X = X.to(self.cfg.DEVICE)
                # 1. 获取预测值与真实值
                d_hat = self.model(X).cpu().numpy()[0]
                d_true = Y_true.numpy()[0]
                
                # 2. 计算预测误差
                metrics["mse"].append(np.mean((d_hat - d_true)**2))
                
                # 3. 决策阶段 (Decision Phase)
                sol_pred = self.solver.solve(d_hat, time_limit=5.0)
                
                # 4. 评估阶段 (Evaluation Phase)
                cost_actual = self.solver.evaluate_true_cost(sol_pred, d_true)
                metrics["true_cost"].append(cost_actual)
                
                # 5. 全知阶段 (Oracle Phase)
                sol_oracle = self.solver.solve(d_true, time_limit=5.0)
                cost_oracle = sol_oracle["obj"]
                metrics["oracle_cost"].append(cost_oracle)
                
                # 6. 计算后悔值
                regret = max(0, cost_actual - cost_oracle)
                metrics["regret"].append(regret)

        # ---------------------------------------------------------
        # 结果统计与持久化保存
        # ---------------------------------------------------------
        avg_mse = np.mean(metrics['mse'])
        avg_true = np.mean(metrics['true_cost'])
        avg_oracle = np.mean(metrics['oracle_cost'])
        avg_regret = np.mean(metrics['regret'])
        relative_gap = (avg_regret / avg_oracle * 100) if avg_oracle > 1e-6 else 0.0

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"result_{timestamp}.txt"
        result_path = os.path.join(self.run_dir, result_filename)

        # 构造结果文本
        result_str = (
            f"Experiment Report - {timestamp}\n"
            f"{'='*40}\n"
            f"Samples Evaluated : {num_samples}\n"
            f"Model Type        : Predict-then-Optimize (PTO)\n"
            f"Vehicle Capacity  : {self.cfg.VEHICLE_CAPACITY}\n"
            f"Backorder Penalty : {self.cfg.BACKORDER_PENALTY}\n"
            f"{'-'*40}\n"
            f"Avg MSE           : {avg_mse:.4f}\n"
            f"Avg True Cost     : {avg_true:.2f}\n"
            f"Avg Oracle Cost   : {avg_oracle:.2f}\n"
            f"Avg Regret        : {avg_regret:.2f}\n"
            f"Relative Gap      : {relative_gap:.2f}%\n"
            f"{'='*40}\n"
        )

        # 打印至控制台
        print(f"\n{result_str}")
        
        # 写入文件
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                f.write(result_str)
            print(f"[Info] Results successfully saved to: {result_path}")
        except Exception as e:
            print(f"[Error] Failed to save results: {e}")

if __name__ == "__main__":
    runner = ExperimentRunner()
    
    # 1. 训练
    runner.train_pto()
    
    # 2. 评估并保存结果
    runner.evaluate(num_samples=20)