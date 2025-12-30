# 文件路径: models/predictor.py

import torch
import torch.nn as nn

class DemandPredictor(nn.Module):
    """
    基于多层感知机 (MLP) 的需求预测模型
    
    架构特征:
    1. 输入层: 接收时间序列的滑动窗口特征。
    2. 隐藏层: 采用双层结构配合 ReLU 激活与 Dropout 正则化，以增强非线性表达能力并防止过拟合。
    3. 输出层: 采用 Softplus 激活函数，从数学上保证输出值恒大于零，符合物理约束。
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        """
        Args:
            input_dim: 输入特征维度 (窗口长度 * 门店数量)
            output_dim: 输出维度 (门店数量)
            hidden_dim: 隐藏层神经元数量
        """
        super(DemandPredictor, self).__init__()
        
        self.net = nn.Sequential(
            # 第一隐藏层
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            
            # 第二隐藏层 (特征压缩)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            # 输出层
            nn.Linear(hidden_dim // 2, output_dim),
            # Softplus(x) = log(1 + exp(x))，确保需求预测值非负
            nn.Softplus() 
        )
        
        # 权重初始化 (He Initialization)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)