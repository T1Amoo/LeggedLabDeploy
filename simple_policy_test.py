#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的策略测试脚本
只做三件事：
1. 生成虚拟观测
2. 输入到策略
3. 打印策略输出的action
"""

import numpy as np
import torch
import argparse
import sys
import os

# 直接导入，因为文件在同一目录
try:
    from dummy_data import DummyDataGenerator
except ImportError:
    # 如果导入失败，直接定义简单的数据生成器
    class DummyDataGenerator:
        def __init__(self, num_actions=29, num_obs=96, history_length=1):
            self.num_actions = num_actions
            self.num_obs = num_obs
            self.history_length = history_length
            np.random.seed(42)
        
        def generate_observation(self):
            """生成96维观测"""
            obs = np.zeros(self.num_obs, dtype=np.float32)
            obs[0:3] = np.random.uniform(-0.1, 0.1, 3)  # 角速度
            obs[3:6] = np.random.uniform(-1, 1, 3)       # 重力方向
            obs[6:9] = np.random.uniform(-0.5, 0.5, 3)   # 命令
            obs[9:9+self.num_actions] = np.random.uniform(-0.5, 0.5, self.num_actions)  # 关节位置
            obs[9+self.num_actions:9+self.num_actions*2] = np.random.uniform(-2.0, 2.0, self.num_actions)  # 关节速度
            obs[9+self.num_actions*2:9+self.num_actions*3] = np.random.uniform(-0.1, 0.1, self.num_actions)  # 之前动作
            return obs

def test_policy_simple(policy_path=None, num_tests=5):
    """简单的策略测试"""
    
    print("=" * 50)
    print("简单策略测试")
    print("=" * 50)
    
    # 1. 创建虚拟数据生成器
    generator = DummyDataGenerator(num_actions=29, num_obs=96, history_length=1)
    
    # 2. 加载策略（如果没有提供策略文件，创建虚拟策略）
    if policy_path and os.path.exists(policy_path):
        try:
            policy = torch.jit.load(policy_path).eval()
            print(f"✓ 成功加载策略: {policy_path}")
        except Exception as e:
            print(f"✗ 加载策略失败: {e}")
            print("创建虚拟策略进行测试...")
            policy = create_dummy_policy()
    else:
        print("没有提供策略文件，创建虚拟策略进行测试...")
        policy = create_dummy_policy()
    
    # 3. 进行多次测试
    for i in range(num_tests):
        print(f"\n--- 测试 {i+1}/{num_tests} ---")
        
        # 生成虚拟观测
        obs = generator.generate_observation()
        print(f"观测维度: {obs.shape}")
        print(f"观测前10个值: {obs[:10]}")
        
        # 准备输入（添加历史维度）
        obs_history = obs.reshape(1, -1)  # 变成 (1, 96)
        obs_tensor = torch.from_numpy(obs_history).float()
        
        # 输入到策略
        with torch.inference_mode():
            action = policy(obs_tensor)
        
        # 打印策略输出
        action_np = action.detach().numpy().squeeze()
        print(f"动作维度: {action_np.shape}")
        print(f"所有动作值:")
        for i, val in enumerate(action_np):
            print(f"  关节{i:2d}: {val:8.4f}")
        print(f"动作范围: [{action_np.min():.4f}, {action_np.max():.4f}]")
        print(f"动作均值: {action_np.mean():.4f}")
        print(f"动作标准差: {action_np.std():.4f}")
        
        # 输出数组格式的数据，方便复制到另一台机器
        print(f"\n=== 数组格式数据 (测试 {i+1}) ===")
        print(f"# 观测数据 (96维)")
        print(f"observation_{i+1} = np.array({obs.tolist()})")
        print(f"# 动作数据 (29维)")
        print(f"action_{i+1} = np.array({action_np.tolist()})")
        print(f"# 观测数据 (Python列表格式)")
        print(f"observation_{i+1}_list = {obs.tolist()}")
        print(f"# 动作数据 (Python列表格式)")
        print(f"action_{i+1}_list = {action_np.tolist()}")
        print("=" * 60)

def create_dummy_policy():
    """创建虚拟策略"""
    class DummyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(96, 64)
            self.fc2 = torch.nn.Linear(64, 29)
            self.activation = torch.nn.ReLU()
            
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = torch.tanh(self.fc2(x)) * 0.5  # 输出范围 [-0.5, 0.5]
            return x
    
    policy = DummyPolicy()
    return policy

def main():
    parser = argparse.ArgumentParser(description='简单策略测试')
    parser.add_argument('--policy_path', type=str, default=None, 
                       help='策略文件路径 (.pt文件)')
    parser.add_argument('--num_tests', type=int, default=5,
                       help='测试次数')
    
    args = parser.parse_args()
    
    # 运行测试
    test_policy_simple(args.policy_path, args.num_tests)
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()
