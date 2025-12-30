"""
主程序入口模块
负责编排数据加载、模型训练、优化求解与结果评估的全流程。
新增功能：自动将实验结果保存至 run 文件夹，并附带时间戳。
"""

import os
import time
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

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

        # 决策导向训练通常需要 batch=1，且只抽样少量样本以控制 MILP 调用次数
        if self.data_processor.train_dataset is not None:
            self.decision_train_loader = DataLoader(
                self.data_processor.train_dataset,
                batch_size=1,
                shuffle=True
            )
        else:
            self.decision_train_loader = None
        
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

        # 6. 初始化结果保存目录（每次运行单独一个子文件夹，避免 run/ 杂乱）
        self.run_base_dir = os.path.join(self.cfg.PROJECT_ROOT, "run")
        os.makedirs(self.run_base_dir, exist_ok=True)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.run_base_dir, self.session_timestamp)
        os.makedirs(self.session_dir, exist_ok=True)

        # 7. 训练过程记录（用于画图）
        self.train_history = {
            "epoch_loss": []
        }

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
            self.train_history["epoch_loss"].append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"[Epoch {epoch+1:02d}/{self.cfg.EPOCHS}] "
                      f"MSE Loss: {avg_loss:.4f} | "
                      f"Time: {time.time()-start_time:.2f}s")
        
        print("Training Completed.")

    def _save_plots(self, metrics: dict, out_dir: str, timestamp: str):
        """将训练/评估的关键曲线保存到 run/ 目录。

        说明：若环境缺少 matplotlib，则自动跳过，不影响主流程。
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[Warning] matplotlib not available, skip saving plots. Details: {e}")
            return

        # 1) 训练损失曲线
        try:
            losses = self.train_history.get("epoch_loss", [])
            if len(losses) > 0:
                fig = plt.figure(figsize=(7, 4))
                plt.plot(range(1, len(losses) + 1), losses)
                plt.xlabel("Epoch")
                plt.ylabel("MSE Loss")
                plt.title("Training Loss (PTO / MSE)")
                plt.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"loss_{timestamp}.png"), dpi=200)
                plt.close(fig)
        except Exception as e:
            print(f"[Warning] Failed to save loss curve: {e}")

        # 2) True vs Oracle cost（样本级对比）
        try:
            true_cost = np.array(metrics.get("true_cost", []), dtype=float)
            oracle_cost = np.array(metrics.get("oracle_cost", []), dtype=float)
            if len(true_cost) > 0 and len(true_cost) == len(oracle_cost):
                fig = plt.figure(figsize=(6, 6))
                plt.scatter(oracle_cost, true_cost, s=18, alpha=0.8)
                min_v = float(min(true_cost.min(), oracle_cost.min()))
                max_v = float(max(true_cost.max(), oracle_cost.max()))
                plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
                plt.xlabel("Oracle Cost")
                plt.ylabel("True Cost (Decision from Prediction)")
                plt.title("Cost Comparison")
                plt.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"cost_scatter_{timestamp}.png"), dpi=200)
                plt.close(fig)
        except Exception as e:
            print(f"[Warning] Failed to save cost scatter plot: {e}")

        # 3) Regret 分布
        try:
            regret = np.array(metrics.get("regret", []), dtype=float)
            if len(regret) > 0:
                fig = plt.figure(figsize=(7, 4))
                plt.hist(regret, bins=min(15, max(5, len(regret))), alpha=0.85)
                plt.xlabel("Regret")
                plt.ylabel("Count")
                plt.title("Regret Distribution")
                plt.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"regret_hist_{timestamp}.png"), dpi=200)
                plt.close(fig)
        except Exception as e:
            print(f"[Warning] Failed to save regret histogram: {e}")

        # 4) 预测需求 vs 真实需求（汇总散点）
        try:
            d_hat_all = np.array(metrics.get("d_hat", []), dtype=float)
            d_true_all = np.array(metrics.get("d_true", []), dtype=float)
            if d_hat_all.size > 0 and d_true_all.size > 0 and d_hat_all.shape == d_true_all.shape:
                fig = plt.figure(figsize=(6, 6))
                plt.scatter(d_true_all.flatten(), d_hat_all.flatten(), s=12, alpha=0.6)
                min_v = float(min(d_true_all.min(), d_hat_all.min()))
                max_v = float(max(d_true_all.max(), d_hat_all.max()))
                plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
                plt.xlabel("True Demand")
                plt.ylabel("Predicted Demand")
                plt.title("Demand Prediction Scatter")
                plt.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"demand_scatter_{timestamp}.png"), dpi=200)
                plt.close(fig)
        except Exception as e:
            print(f"[Warning] Failed to save demand scatter plot: {e}")

        # 5) 每个门店单独的预测-真实曲线（更适合作业报告展示）
        try:
            d_hat_all = np.array(metrics.get("d_hat", []), dtype=float)
            d_true_all = np.array(metrics.get("d_true", []), dtype=float)
            if d_hat_all.size > 0 and d_true_all.size > 0 and d_hat_all.shape == d_true_all.shape:
                num_samples = d_hat_all.shape[0]
                num_stores = d_hat_all.shape[1]
                x = np.arange(1, num_samples + 1)

                for store_idx in range(num_stores):
                    fig = plt.figure(figsize=(8, 4))
                    plt.plot(x, d_true_all[:, store_idx], label="True", linewidth=2)
                    plt.plot(x, d_hat_all[:, store_idx], label="Pred", linewidth=2)
                    plt.xlabel("Sample Index")
                    plt.ylabel("Weekly Demand")
                    plt.title(f"Store {store_idx + 1}: True vs Pred")
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    fig.tight_layout()
                    fig.savefig(
                        os.path.join(out_dir, f"store_{store_idx + 1}_curve_{timestamp}.png"),
                        dpi=200
                    )
                    plt.close(fig)
        except Exception as e:
            print(f"[Warning] Failed to save per-store curves: {e}")

        print(f"[Info] Plots saved to: {out_dir}")

    def evaluate(self, num_samples: int = 20, tag: str = "baseline"):
        """
        全流程评估与结果记录
        
        Args:
            num_samples: 评估样本数量 (默认20以节省时间)
        """
        # num_samples <= 0 代表评估全部测试集
        if num_samples is None or num_samples <= 0:
            num_samples = len(self.test_loader)

        print(f"\n{'='*20} Start Evaluation (Top {num_samples} Samples) {'='*20}")
        self.model.eval()
        
        metrics = {
            "mse": [],
            "true_cost": [],
            "oracle_cost": [],
            "regret": [],
            "d_hat": [],
            "d_true": []
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

                metrics["d_hat"].append(d_hat)
                metrics["d_true"].append(d_true)
                
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

        # 生成带时间戳的文件名，并按 tag 分目录保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tag = (tag or "baseline").strip().lower()
        out_dir = os.path.join(self.session_dir, safe_tag)
        os.makedirs(out_dir, exist_ok=True)

        result_filename = f"result_{timestamp}.txt"
        result_path = os.path.join(out_dir, result_filename)

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

        # 保存图像
        self._save_plots(metrics, out_dir, timestamp)

        return {
            "avg_mse": float(avg_mse),
            "avg_true": float(avg_true),
            "avg_oracle": float(avg_oracle),
            "avg_regret": float(avg_regret),
            "relative_gap": float(relative_gap),
            "timestamp": timestamp,
            "result_path": result_path,
        }

    def train_decision_aware(
        self,
        epochs: int = 5,
        samples_per_epoch: int = 30,
        epsilon: float = 5.0,
        milp_time_limit: float = 1.0,
    ):
        """面向决策的训练：用“真实成本(基于MILP决策+真实需求评估)”做目标。

        由于 MILP 不可微，这里用有限差分在模型输出 d_hat 上近似 cost 的梯度，
        再把该梯度作为 pseudo-gradient 反传到网络参数。

        该方法计算开销较大，默认只抽样少量训练样本，并限制 MILP time_limit。
        """
        if self.decision_train_loader is None:
            raise RuntimeError("Decision-aware training requires access to train_dataset.")

        print(f"\n{'='*20} Start Decision-Aware Training (Finite Difference on True Cost) {'='*20}")
        print(
            f"[Config] epochs={epochs}, samples_per_epoch={samples_per_epoch}, "
            f"epsilon={epsilon}, milp_time_limit={milp_time_limit}s"
        )
        self.model.train()

        for epoch in range(epochs):
            start_time = time.time()
            seen = 0
            avg_costs = []

            for X, Y_true in self.decision_train_loader:
                if seen >= samples_per_epoch:
                    break

                X = X.to(self.cfg.DEVICE)
                d_true = Y_true.numpy()[0]

                # forward
                preds = self.model(X)[0]  # shape: (num_stores,)
                d_hat = preds.detach().cpu().numpy()

                # base decision & cost
                sol_base = self.solver.solve(d_hat, time_limit=milp_time_limit)
                base_cost = self.solver.evaluate_true_cost(sol_base, d_true)

                # finite-diff gradient w.r.t d_hat
                grad = np.zeros_like(d_hat, dtype=float)
                for k in range(len(d_hat)):
                    d_hat_perturb = d_hat.copy()
                    d_hat_perturb[k] = max(0.0, float(d_hat_perturb[k] + epsilon))
                    sol_p = self.solver.solve(d_hat_perturb, time_limit=milp_time_limit)
                    cost_p = self.solver.evaluate_true_cost(sol_p, d_true)
                    grad[k] = (cost_p - base_cost) / float(epsilon)

                # backprop with custom gradient
                self.optimizer.zero_grad()
                grad_t = torch.tensor(grad, dtype=preds.dtype, device=self.cfg.DEVICE)
                preds.backward(gradient=grad_t)
                self.optimizer.step()

                avg_costs.append(float(base_cost))
                seen += 1

            mean_cost = float(np.mean(avg_costs)) if len(avg_costs) else float('inf')
            print(
                f"[Epoch {epoch+1:02d}/{epochs}] "
                f"Samples: {seen} | "
                f"Mean TrueCost(base): {mean_cost:.2f} | "
                f"Time: {time.time()-start_time:.2f}s"
            )

        print("Decision-aware training completed.")

    def compare_models(
        self,
        eval_samples: int = 20,
        decision_epochs: int = 5,
        decision_samples_per_epoch: int = 30,
        decision_epsilon: float = 5.0,
        decision_milp_time_limit: float = 1.0,
    ):
        """对比：MSE 基线 vs 面向决策训练（同一测试集，以真实成本为主指标）。"""
        # 1) Baseline: MSE
        print(f"\n{'='*20} Baseline: MSE Training {'='*20}")
        self.train_pto()
        baseline_res = self.evaluate(num_samples=eval_samples, tag="baseline")

        # 2) Decision-aware: re-init model & optimizer
        print(f"\n{'='*20} Decision-Aware Model {'='*20}")
        self.model = DemandPredictor(
            input_dim=self.input_dim,
            output_dim=self.cfg.NUM_STORES,
            hidden_dim=self.cfg.HIDDEN_DIM
        ).to(self.cfg.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LEARNING_RATE)
        self.train_history = {"epoch_loss": []}

        self.train_decision_aware(
            epochs=decision_epochs,
            samples_per_epoch=decision_samples_per_epoch,
            epsilon=decision_epsilon,
            milp_time_limit=decision_milp_time_limit,
        )
        decision_res = self.evaluate(num_samples=eval_samples, tag="decision")

        # 3) Save comparison summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_path = os.path.join(self.session_dir, f"comparison_{timestamp}.txt")
        comp_str = (
            f"Comparison Report - {timestamp}\n"
            f"{'='*50}\n"
            f"Test Samples     : {eval_samples}\n"
            f"Metric           : True Cost (lower is better)\n"
            f"{'-'*50}\n"
            f"Baseline (MSE)   : AvgTrue={baseline_res['avg_true']:.2f}, AvgOracle={baseline_res['avg_oracle']:.2f}, Gap={baseline_res['relative_gap']:.2f}%\n"
            f"Decision-Aware   : AvgTrue={decision_res['avg_true']:.2f}, AvgOracle={decision_res['avg_oracle']:.2f}, Gap={decision_res['relative_gap']:.2f}%\n"
            f"{'='*50}\n"
        )
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write(comp_str)
        print(f"[Info] Comparison saved to: {comp_path}")

        # optional bar plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(7, 4))
            labels = ["Baseline(MSE)", "Decision-Aware"]
            values = [baseline_res["avg_true"], decision_res["avg_true"]]
            plt.bar(labels, values)
            plt.ylabel("Avg True Cost")
            plt.title("Model Comparison on True Cost")
            fig.tight_layout()
            fig.savefig(os.path.join(self.session_dir, f"comparison_bar_{timestamp}.png"), dpi=200)
            plt.close(fig)
        except Exception as e:
            print(f"[Warning] Failed to save comparison bar plot: {e}")

        return baseline_res, decision_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "decision", "compare"],
        help="baseline: MSE training; decision: decision-aware training; compare: run both and compare"
    )
    # Defaults pinned to the last-used experiment settings
    parser.add_argument("--eval-samples", type=int, default=50, help="<=0 means evaluate full test set")
    parser.add_argument("--decision-epochs", type=int, default=8)
    parser.add_argument("--decision-samples-per-epoch", type=int, default=80)
    parser.add_argument("--decision-epsilon", type=float, default=20.0)
    parser.add_argument("--decision-milp-time-limit", type=float, default=2.5)
    args = parser.parse_args()

    runner = ExperimentRunner()

    if args.mode == "baseline":
        runner.train_pto()
        runner.evaluate(num_samples=args.eval_samples, tag="baseline")
    elif args.mode == "decision":
        runner.train_decision_aware(
            epochs=args.decision_epochs,
            samples_per_epoch=args.decision_samples_per_epoch,
            epsilon=args.decision_epsilon,
            milp_time_limit=args.decision_milp_time_limit,
        )
        runner.evaluate(num_samples=args.eval_samples, tag="decision")
    else:
        runner.compare_models(
            eval_samples=args.eval_samples,
            decision_epochs=args.decision_epochs,
            decision_samples_per_epoch=args.decision_samples_per_epoch,
            decision_epsilon=args.decision_epsilon,
            decision_milp_time_limit=args.decision_milp_time_limit,
        )