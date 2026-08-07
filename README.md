# generals_agent

用强化学习（PPO + self-play）训练 [generals.io](https://generals.io)（军棋/领土扩张实时策略游戏）的 agent。

基于 [strakam/generals-bots](https://github.com/strakam/generals-bots) 的 numpy Gymnasium 环境（完整实现战争迷雾、城堡占领、山脉、战斗结算、兵力增长），参考该仓库论文 *"Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning"* (arXiv:2507.06825) 的势能型奖励塑形方法。

## 环境要求

- Windows + Anaconda（或任意 Python 3.12）
- RTX 50 系（Blackwell）GPU 需要 CUDA 12.8 的 PyTorch

## 安装

```bash
# 1. 创建 conda 环境
conda create -n generals python=3.12 -y
conda activate generals

# 2. 安装依赖
pip install torch --index-url https://download.pytorch.org/whl/cu128   # 非 50 系显卡可省略，直接 pip install torch
pip install generals-bots==2.5.0 gymnasium tensorboard
```

## 快速验证环境

```bash
python verify_env.py
```

应输出 obs/mask 形状、随机对局结果和奖励计算样例。

## 训练

```bash
# 冒烟测试（小地图、少量环境，验证训练管线）
python train/train.py --num-envs 8 --num-steps 32 --iterations 20 --grid-min 6 --grid-max 6 --no-selfplay

# 正式训练（配置在 config.py，默认开启地图尺寸课程 + self-play）
python train/train.py

# 参数覆盖示例
python train/train.py --num-envs 64 --iterations 5000 --run-name grid18

# 关闭课程学习（固定地图）
python train/train.py --no-curriculum
python train/train.py --grid-min 12 --grid-max 14   # 传地图参数时自动退化为固定地图
```

训练指标（胜率、熵、奖励、SPS）写入 TensorBoard：

```bash
tensorboard --logdir runs
```

检查点保存在 `checkpoints/`（`model_<iter>.pt`、`latest.pt`、`best.pt`）。

## 评估与观看

```bash
# 对内置 bot 的胜率评估
python -c "import torch, config; from agents.network import PolicyValueNetwork; from train.eval import evaluate; \
net = PolicyValueNetwork(config.INPUT_CHANNELS, config.HIDDEN_CHANNELS, config.PAD_TO, config.VALUE_HIDDEN); \
net.load_state_dict(torch.load('checkpoints/latest.pt', map_location='cpu')['model_state']); \
print(evaluate(net, config, num_games=20, device='cpu'))"

# GUI 观看 agent 对战 bot
python play.py --model checkpoints/latest.pt --opponent expander
```

## 项目结构

```
config.py               # 全部超参数（网格、PPO、self-play、奖励权重）
agents/
  network.py            # PyTorch CNN 策略-价值网络 + 动作编解码
  rewards.py            # 势能型复合奖励（兵力比 / 地盘比 / 城堡数）
  ppo_agent.py          # 把 PyTorch 策略封装成 generals Agent（自对弈/评估用）
  generals_env.py       # 单玩家 gym 包装器（对手由外部控制）
train/
  train.py              # PPO 训练主循环（CleanRL 风格 + self-play）
  eval.py               # 对内置 bot 的胜率评估
  self_play.py          # 对手池（保存历史策略快照）
play.py                 # GUI 观看 agent 对局
verify_env.py           # 环境冒烟测试
```

## 训练策略

- **奖励**：非终止步使用势能差塑形（`potential(s') - potential(s)`，含兵力比、地盘比、城堡数），终止步给 +1/-1
- **Self-play**：对手池保留最近 N 个策略快照，环境每次重置时采样新对手（可混入 RandomAgent / ExpanderAgent 作为课程前期对手）
- **地图尺寸课程**：默认分阶段 8×8 → 10×12 → 12×14。每阶段对当前地图上 bot 的贪婪评估胜率达到阈值即晋级（或达到该阶段迭代数上限防止卡死），网络按最大地图固定尺寸，跨阶段断点兼容。见 `config.py` 的 `CURRICULUM`
- **动作空间**：每 tick 从 9 通道动作（4 方向 × 全量/半量 + 跳过）中采样，用 `compute_valid_move_mask` 屏蔽非法移动

## 后续路线

- numpy 环境训练收敛慢时可迁移到 generals-bots 的 [JAX 版](https://github.com/strakam/generals-bots)（快 100 倍以上）
- 训练出强 agent 后可用 `generals.remote.generalsio_client` 的 autopilot 部署到 generals.io 真实服务器对战真人
