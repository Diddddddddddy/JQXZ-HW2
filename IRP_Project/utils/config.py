"""
全局配置模块
该模块定义了项目所需的物理参数、训练超参数及文件路径配置。
采用 Dataclass 确保配置的不可变性与类型安全。
"""

from dataclasses import dataclass
import os
import torch

@dataclass
class Config:
    # ---------------------------------------------------------
    # 1. 路径配置 (Path Configuration)
    # ---------------------------------------------------------
    # 动态获取项目根目录：当前文件 (utils/config.py) 的上两级目录
    # 适配路径: C:\Users\kelvin\...\IRP_Project
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 数据目录结构
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    RAW_DATA_PATH: str = os.path.join(DATA_DIR, "raw", "train.csv")
    PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
    TOPOLOGY_DIR: str = os.path.join(DATA_DIR, "topology")
    
    # ---------------------------------------------------------
    # 2. 物理环境与优化参数 (Physics & Optimization Constraints)
    # ---------------------------------------------------------
    # 实验设置
    NUM_STORES: int = 5            # 实验选用的门店数量 (建议保持在 5-10 以确保求解效率)
    TARGET_ITEM_ID: int = 1        # 选定优化的单一商品ID
    
    # 约束参数
    VEHICLE_CAPACITY: float = 200.0  # 车辆最大载重 (Q_max)
    
    # 成本参数
    # 缺货惩罚 (b_j): 建议设定较高数值 (如 20.0-50.0)，以凸显准确预测对避免缺货的重要性
    BACKORDER_PENALTY: float = 20.0  
    TRANSPORT_UNIT_COST: float = 1.0 # 单位距离运输成本 (c_ij 系数)
    
    # 拓扑生成参数
    MAP_SIZE: int = 100            # 模拟地图的边长 (100x100)
    RANDOM_SEED: int = 42          # 随机种子，确保每次生成的门店坐标一致

    # ---------------------------------------------------------
    # 3. 数据与特征工程参数 (Data & Feature Engineering)
    # ---------------------------------------------------------
    HISTORY_WINDOW: int = 4        # 滑动窗口大小：使用过去 N 周的数据预测下一周
    TRAIN_RATIO: float = 0.8       # 训练集占比 (剩余 0.2 为测试集)
    BATCH_SIZE: int = 32           # 数据加载的批次大小
    
    # ---------------------------------------------------------
    # 4. 深度学习训练参数 (Deep Learning Hyperparameters)
    # ---------------------------------------------------------
    LEARNING_RATE: float = 1e-3    # 学习率
    EPOCHS: int = 50               # 训练轮次
    HIDDEN_DIM: int = 64           # 神经网络隐藏层维度
    
    # 自动检测计算设备 (优先使用 GPU)
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """
        初始化后的钩子函数
        自动检测并创建必要的目录结构，防止因目录缺失导致的 IO 错误。
        """
        dirs_to_create = [
            self.DATA_DIR,
            os.path.dirname(self.RAW_DATA_PATH),
            self.PROCESSED_DATA_DIR,
            self.TOPOLOGY_DIR
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            # print(f"[Config] Verified directory: {directory}")