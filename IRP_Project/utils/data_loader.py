"""
数据加载与处理模块
负责原始数据的聚合、特征提取、张量转换及物流拓扑生成。
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import os
from .config import Config

class IRPDataset(Dataset):
    """
    自定义 PyTorch 数据集
    将特征矩阵 X 与 真实需求标签 Y 封装为张量对。
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.X = torch.FloatTensor(features)
        self.Y = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class DataProcessor:
    """
    数据处理核心类
    包含：拓扑生成、时序聚合、特征工程
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.dist_matrix = None
        self.store_coords = None

    def generate_topology(self) -> np.ndarray:
        """
        生成物流网络拓扑
        1. 设定随机种子确保可复现。
        2. 生成 (Num_Stores + 1) 个节点的坐标，索引0为仓库，1-N为门店。
        3. 计算欧氏距离矩阵。
        
        Returns:
            dist_matrix (np.ndarray): (N+1) x (N+1) 距离矩阵
        """
        np.random.seed(self.cfg.RANDOM_SEED)
        
        # 生成坐标: [Depot, Store_1, ..., Store_N]
        # 范围在 [0, MAP_SIZE]
        self.store_coords = np.random.rand(self.cfg.NUM_STORES + 1, 2) * self.cfg.MAP_SIZE
        
        # 初始化距离矩阵
        num_nodes = self.cfg.NUM_STORES + 1
        self.dist_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist = np.linalg.norm(self.store_coords[i] - self.store_coords[j])
                self.dist_matrix[i, j] = dist
                
        print(f"[Info] Topology generated. Depot location: {self.store_coords[0]}")
        return self.dist_matrix

    def _feature_engineering(self, df_pivot: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        特征工程实现
        输入: 透视后的周销量数据 (Index=Week, Columns=Stores)
        输出: 特征矩阵 X, 标签矩阵 Y
        
        特征策略:
        - 针对第 t 周的预测，使用 t-window 到 t-1 周的历史销量作为特征。
        - 扁平化处理：将所有门店的历史数据展平为一维向量作为输入特征。
        """
        data = df_pivot.values # Shape: (Num_Weeks, Num_Stores)
        num_weeks = data.shape[0]
        window = self.cfg.HISTORY_WINDOW
        
        X, Y = [], []
        
        # 这是一个多对多的预测：输入过去N周所有门店的数据，预测当前周所有门店的需求
        for i in range(window, num_weeks):
            # 特征: 过去 window 周的数据，展平
            # shape: (window, num_stores) -> (window * num_stores, )
            feature_vec = data[i-window:i, :].flatten()
            
            # 标签: 当前第 i 周的真实需求
            target_vec = data[i, :]
            
            X.append(feature_vec)
            Y.append(target_vec)
            
        return np.array(X), np.array(Y)

    def load_data(self) -> Tuple[DataLoader, DataLoader, int]:
        """
        主数据加载流程
        1. 读取CSV
        2. 过滤指定Item和Stores
        3. 按周(W)聚合
        4. 构建特征
        5. 划分训练/测试集
        """
        # 1. 检查数据是否存在
        if not os.path.exists(self.cfg.RAW_DATA_PATH):
            raise FileNotFoundError(f"Raw data not found at {self.cfg.RAW_DATA_PATH}. Please check if train.csv is in data/raw/.")

        print(f"[Info] Loading real data from {self.cfg.RAW_DATA_PATH}...")
        df = pd.read_csv(self.cfg.RAW_DATA_PATH)
        
        # 2. 数据转换与筛选
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        
        # 筛选特定商品和门店子集 (Store ID 从 1 开始)
        selected_stores = list(range(1, self.cfg.NUM_STORES + 1))
        # 过滤数据：只保留我们关心的 Item ID 和 Store IDs
        mask = (df['item'] == self.cfg.TARGET_ITEM_ID) & (df['store'].isin(selected_stores))
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            raise ValueError("Filtered data is empty. Please check TARGET_ITEM_ID or NUM_STORES in config.")

        # 3. 按周聚合 (Resample)
        # 将数据透视为：索引=日期，列=门店，值=销量
        df_filtered.set_index('date', inplace=True)
        df_pivot = df_filtered.pivot(columns='store', values='sales')
        
        # 按周求和 (W = Weekly, ending Sunday)
        df_weekly = df_pivot.resample('W').sum().fillna(0)
        
        print(f"[Info] Data aggregated to weekly level. Shape: {df_weekly.shape}")

        # 4. 特征工程
        features, targets = self._feature_engineering(df_weekly)
        
        # 5. 划分数据集
        dataset_size = len(features)
        train_size = int(dataset_size * self.cfg.TRAIN_RATIO)
        
        X_train, Y_train = features[:train_size], targets[:train_size]
        X_test, Y_test = features[train_size:], targets[train_size:]
        
        # 标准化 (Scaling) - 仅对输入特征标准化，Label保持原始量级以便计算真实成本
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        # 6. 封装 DataLoader
        train_dataset = IRPDataset(X_train, Y_train)
        test_dataset = IRPDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.cfg.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # 测试集batch=1便于逐个评估
        
        input_dim = X_train.shape[1]
        
        print(f"[Info] Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        return train_loader, test_loader, input_dim