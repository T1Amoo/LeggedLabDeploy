# 修改总结

## 主要修改内容

### 1. 自动关节识别
- **之前**：需要手动指定 `--upper_body_joints` 参数
- **现在**：根据机器人类型自动识别上半身和下半身关节

#### G1机器人 (msg_type: "hg")
- 上半身关节：22-28 (7个关节)
  - 22: R_SHOULDER_PITCH
  - 23: R_SHOULDER_ROLL  
  - 24: R_SHOULDER_YAW
  - 25: R_ELBOW
  - 26: R_WRIST_ROLL
  - 27: R_WRIST_PITCH
  - 28: R_WRIST_YAW
- 下半身关节：0-21 (22个关节)
  - 0-11: 左腿关节 (L_LEG_HIP_PITCH 到 L_LEG_ANKLE_ROLL)
  - 12-14: 腰部关节 (WAIST_YAW, WAIST_ROLL, WAIST_PITCH)
  - 15-21: 左臂关节 (L_SHOULDER_PITCH 到 L_WRIST_YAW)

#### H1机器人 (msg_type: "go")
- 上半身关节：18-19 (2个关节)
  - 18: L_ELBOW
  - 19: R_ELBOW
- 下半身关节：0-17 (18个关节)
  - 0-17: 腿部关节 (L_LEG_HIP_PITCH 到 R_LEG_ANKLE_ROLL)

### 2. 简化的命令行接口
```bash
# 之前
python deploy_hybrid.py \
    --config_path configs/g1.yaml \
    --upper_body_csv path/to/trajectory.csv \
    --upper_body_joints 22 23 24 25 26 27 28 \
    --net eno1

# 现在
python deploy_hybrid.py \
    --config_path configs/g1.yaml \
    --upper_body_csv path/to/trajectory.csv \
    --net eno1
```

### 3. 自动关节映射
```python
# 自动识别逻辑
if config.msg_type == "hg":  # G1 robot
    self.upper_body_indices = list(range(22, config.num_actions))
    self.lower_body_indices = list(range(0, 22))
elif config.msg_type == "go":  # H1 robot
    self.upper_body_indices = list(range(18, config.num_actions))
    self.lower_body_indices = list(range(0, 18))
```

## 文件修改列表

### 1. `deploy_hybrid.py`
- 移除 `upper_body_joints` 参数
- 添加自动关节识别逻辑
- 更新构造函数和初始化代码
- 简化命令行参数解析

### 2. `README_hybrid.md`
- 更新使用说明
- 移除手动关节配置说明
- 添加自动识别说明
- 更新示例命令

### 3. 新增文件
- `test_auto_joints.py`: 测试自动关节识别功能
- `trajectories/g1_arms_example.csv`: G1机器人示例轨迹
- `trajectories/h1_arms_example.csv`: H1机器人示例轨迹

## 使用优势

1. **简化使用**：无需手动指定关节索引
2. **减少错误**：避免手动配置错误
3. **自动适配**：支持不同机器人类型
4. **向后兼容**：保持原有功能不变

## 注意事项

1. CSV文件列数必须与机器人上半身关节数匹配
   - G1: 7列
   - H1: 2列
2. 关节索引现在完全自动化，无需手动配置
3. 系统会根据配置文件自动选择正确的关节映射

## 测试验证

运行测试脚本验证功能：
```bash
python test_auto_joints.py
```

这将验证：
- G1机器人关节识别
- H1机器人关节识别  
- CSV验证逻辑
