"""
MILP 优化求解器模块
负责构建、求解库存路径问题 (IRP) 的数学模型，并评估决策的真实成本。
"""

import numpy as np
from typing import Dict, Any

# 尝试导入 Gurobi，若未安装则抛出警告
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("[Warning] 'gurobipy' not found. Optimization functionality will fail.")

from utils.config import Config

class IRPSolver:
    """
    库存路径问题 (IRP) 求解器
    基于 Gurobi 实现混合整数线性规划模型。
    """
    def __init__(self, config: Config, dist_matrix: np.ndarray):
        """
        初始化求解器
        Args:
            config: 全局配置对象
            dist_matrix: (N+1)x(N+1) 距离矩阵，节点0为仓库
        """
        self.cfg = config
        self.dist_matrix = dist_matrix
        
        # 节点定义
        # nodes: 所有节点 [0, 1, ..., N]
        # stores: 仅门店节点 [1, ..., N]
        self.num_nodes = self.dist_matrix.shape[0]
        self.nodes = list(range(self.num_nodes))
        self.stores = list(range(1, self.num_nodes))

        if not HAS_GUROBI:
            raise RuntimeError("Gurobi is required to initialize IRPSolver.")

    def solve(self, predicted_demand: np.ndarray, time_limit: float = 10.0) -> Dict[str, Any]:
        """
        核心方法：基于预测需求构建并求解 MILP 模型。
        
        Args:
            predicted_demand: 维度为 (Num_Stores,) 的预测需求向量
            time_limit: 求解器时间限制 (秒)
            
        Returns:
            solution (Dict): 包含配送量 'q'、路径 'z'、目标值 'obj' 及状态 'status'
        """
        # 1. 创建模型环境
        m = gp.Model("IRP_Weekly")
        
        # 关闭求解日志输出，保持控制台整洁
        m.setParam('OutputFlag', 0)
        m.setParam('TimeLimit', time_limit)

        # -------------------------------------------------------
        # 2. 定义决策变量
        # -------------------------------------------------------
        # z[i,j]: 二元变量，车辆是否从 i 直接行驶到 j
        z = m.addVars(self.nodes, self.nodes, vtype=GRB.BINARY, name="z")
        
        # q[j]: 连续变量，向门店 j 配送的数量
        q = m.addVars(self.stores, lb=0.0, ub=self.cfg.VEHICLE_CAPACITY, name="q")
        
        # B[j]: 连续变量，门店 j 的缺货量 (Backorder)
        # 在优化模型中，这是基于预测需求的“预期缺货量”
        B = m.addVars(self.stores, lb=0.0, name="B")
        
        # u[i]: 连续变量，用于 MTZ 子回路消除约束的辅助变量
        # 代表车辆访问节点 i 时的累积载荷或顺序
        u = m.addVars(self.stores, lb=0.0, ub=self.cfg.VEHICLE_CAPACITY, name="u")

        # -------------------------------------------------------
        # 3. 构建目标函数 (Minimize Predicted Cost)
        # -------------------------------------------------------
        # 运输成本: 距离 * 单位运费
        transport_cost = gp.quicksum(
            self.dist_matrix[i, j] * z[i, j] 
            for i in self.nodes for j in self.nodes if i != j
        ) * self.cfg.TRANSPORT_UNIT_COST
        
        # 缺货惩罚: 预期缺货量 * 单位惩罚
        shortage_cost = gp.quicksum(
            B[j] * self.cfg.BACKORDER_PENALTY 
            for j in self.stores
        )
        
        m.setObjective(transport_cost + shortage_cost, GRB.MINIMIZE)

        # -------------------------------------------------------
        # 4. 添加约束条件
        # -------------------------------------------------------
        
        # (1) 流量守恒 (Flow Balance)
        # 仓库: 出度=1, 入度=1 (假设单辆车必须出发并返回)
        m.addConstr(gp.quicksum(z[0, j] for j in self.stores) == 1, "DepotOut")
        m.addConstr(gp.quicksum(z[i, 0] for i in self.stores) == 1, "DepotIn")
        
        # 门店: 流入 = 流出
        for j in self.stores:
            m.addConstr(
                gp.quicksum(z[i, j] for i in self.nodes if i != j) == 
                gp.quicksum(z[j, k] for k in self.nodes if k != j), 
                f"FlowBalance_{j}"
            )
            
            # 每个门店最多被访问一次
            m.addConstr(
                gp.quicksum(z[i, j] for i in self.nodes if i != j) <= 1, 
                f"VisitOnce_{j}"
            )

        # (2) 车辆容量约束 (Capacity)
        m.addConstr(
            gp.quicksum(q[j] for j in self.stores) <= self.cfg.VEHICLE_CAPACITY, 
            "TotalCapacity"
        )

        # (3) 需求满足平衡 (Demand Satisfaction)
        # 配送量 + 缺货量 >= 预测需求
        # 注意：此处使用 >= 配合最小化目标函数，求解器会自动让等式成立以减少成本
        for idx, j in enumerate(self.stores):
            d_hat = predicted_demand[idx]
            # 必须处理负数预测值 (虽然ReLU已保证非负，但增加鲁棒性)
            d_hat = max(0.0, float(d_hat)) 
            m.addConstr(q[j] + B[j] >= d_hat, f"DemandSat_{j}")

        # (4) 逻辑关联: 未访问则无法配送
        # q_j <= Q_max * y_j (其中 y_j 为是否访问 j)
        for j in self.stores:
            is_visited = gp.quicksum(z[i, j] for i in self.nodes if i != j)
            m.addConstr(q[j] <= self.cfg.VEHICLE_CAPACITY * is_visited, f"LinkService_{j}")

        # (5) 子回路消除 (MTZ Constraints)
        # u_i - u_j + Q * z_ij <= Q - q_j
        for i in self.stores:
            for j in self.stores:
                if i != j:
                    m.addConstr(
                        u[i] - u[j] + self.cfg.VEHICLE_CAPACITY * z[i, j] <= 
                        self.cfg.VEHICLE_CAPACITY - q[j], 
                        f"MTZ_{i}_{j}"
                    )

        # -------------------------------------------------------
        # 5. 求解与结果解析
        # -------------------------------------------------------
        m.optimize()

        res = {
            "status": m.status,
            "obj": float('inf'),
            "q": np.zeros(len(self.stores)),
            "routes": []
        }

        if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
            res["obj"] = m.objVal
            # 提取配送量
            res["q"] = np.array([q[j].X for j in self.stores])
            
            # 提取路径 (解析 z 变量)
            active_arcs = []
            for i in self.nodes:
                for j in self.nodes:
                    if i != j and z[i, j].X > 0.5:
                        active_arcs.append((i, j))
            res["routes"] = active_arcs
        
        m.dispose()
        return res

    def evaluate_true_cost(self, decision: Dict[str, Any], true_demand: np.ndarray) -> float:
        """
        核心评估函数：计算真实成本 (True Cost)
        
        逻辑:
        1. 运输成本: 由决策中的路径 (z) 决定，是固定的。
        2. 缺货/持有成本: 由决策中的配送量 (q) 与 真实需求 (d) 的差异决定。
        
        公式: Cost = Transport + Penalty * [d_true - q_dec]_+
        
        Args:
            decision: solve() 方法返回的决策字典
            true_demand: 真实需求向量
            
        Returns:
            true_cost: 浮点数值
        """
        # 若求解失败，返回极大的惩罚值
        if decision["status"] not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            return 1e6

        # 1. 计算实际运输成本
        transport_cost = 0.0
        for (i, j) in decision["routes"]:
            transport_cost += self.dist_matrix[i, j] * self.cfg.TRANSPORT_UNIT_COST
            
        # 2. 计算实际缺货成本 (Inventory/Shortage Cost)
        shortage_cost = 0.0
        decision_q = decision["q"]
        
        for idx in range(len(self.stores)):
            d_real = true_demand[idx]
            q_val = decision_q[idx]
            
            # 计算缺货量: max(0, 需求 - 配送)
            shortage = max(0.0, d_real - q_val)
            
            shortage_cost += shortage * self.cfg.BACKORDER_PENALTY
            
        total_true_cost = transport_cost + shortage_cost
        return total_true_cost