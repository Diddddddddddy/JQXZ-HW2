# IRP Project（方向B：Predict-then-Optimize）

本项目实现了一个简化的库存路径问题（IRP, Inventory Routing Problem）的 **Predict-then-Optimize (PTO)** 流程：

1. 从历史销量构造滑动窗口特征，训练需求预测模型（MLP）。
2. 使用预测需求作为输入，构建并求解 IRP 的 MILP 模型（Gurobi）。
3. 用真实需求评估“预测驱动的决策”成本，并与“全知（Oracle）决策”对比，计算 regret。

## 目录结构

- `main.py`：实验入口（训练 + 评估 + 结果保存）
- `utils/config.py`：全局参数与路径配置
- `utils/data_loader.py`：数据聚合、特征工程、DataLoader、随机拓扑生成
- `models/predictor.py`：需求预测模型（MLP + Softplus 输出）
- `optimization/milp_solver.py`：IRP 的 MILP 建模与求解、真实成本评估
- `data/raw/train.csv`：原始数据（作业提供）
- `run/`：运行后自动生成的结果文件夹（带时间戳的 `.txt` 报告）

## 环境与依赖

- Python 3.9+（建议 3.10/3.11）
- 主要依赖见 `requirements.txt`
- **可选/必需（用于优化求解）**：Gurobi + `gurobipy`
  - 若未安装 `gurobipy`，项目会在初始化优化器时给出提示并退出。

## 数据放置

确保数据文件存在：

- `data/raw/train.csv`

项目会自动创建以下目录（若不存在）：

- `data/raw/`
- `data/processed/`
- `data/topology/`

## 运行方式

在 `IRP_Project/` 目录下执行：

```bash
python main.py
```

也可以选择不同模式（用于作业对比实验）：

```bash
# 仅基线：MSE (Predict-then-Optimize)
python main.py --mode baseline

# 仅面向决策训练（有限差分近似真实成本梯度）
python main.py --mode decision

# 对比：基线 vs 面向决策（同一测试集以真实成本比较）
python main.py --mode compare

# 评估更多样本（<=0 表示评估整个测试集）
python main.py --mode compare --eval-samples 0

# 说明：当前代码已将默认参数固定为最近一次实验使用的设置：
# eval-samples=50, decision-epochs=8, decision-samples-per-epoch=80, decision-epsilon=20, decision-milp-time-limit=2.5
# 仍可通过命令行覆盖默认值。

# 调大决策导向训练预算（更可能接近/超过 baseline，但会更慢）
python main.py --mode compare --eval-samples 0 --decision-epochs 10 --decision-samples-per-epoch 200 --decision-epsilon 20 --decision-milp-time-limit 2.5
```

运行结束后会在 `run/` 下生成类似以下文件：

每次运行会创建一个独立子目录：

- `run/<run_timestamp>/baseline/`（或 `decision/`）

其中包含：

- `result_YYYYMMDD_HHMMSS.txt`

并保存图像（若环境中安装了 `matplotlib`）：

输出文件均位于本次运行目录 `run/<run_timestamp>/` 下：

- `baseline/` 或 `decision/`：评估报告与图像
- `comparison_*.txt` / `comparison_bar_*.png`：对比实验输出（`--mode compare`）

其中包含：平均 MSE、平均真实成本、平均 Oracle 成本、平均 regret、相对 gap 等。

## 常见问题

### 1) 提示缺少 gurobipy / Gurobi

`optimization/milp_solver.py` 使用 Gurobi 求解 MILP。

- 需要你本机安装 Gurobi 并正确配置许可证
- 再安装 Python 接口 `gurobipy`

若学校/作业环境不提供 Gurobi，建议在报告中说明求解器依赖（并展示你完成了预测与 MILP 建模代码）。
