# generals_agent

用强化学习（PPO + self-play）训练 [generals.io](https://generals.io)（军棋/领土扩张实时策略游戏）的 agent。

基于 [strakam/generals-bots](https://github.com/strakam/generals-bots) 的 numpy Gymnasium 环境（完整实现战争迷雾、城堡占领、山脉、战斗结算、兵力增长），参考该仓库论文 *"Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning"* (arXiv:2507.06825) 的势能型奖励塑形与**记忆增强观测**方法。

> ⚠️ **检查点不兼容**：本次升级把观测从 15 通道改为 **31 通道**（记忆增强），旧 `checkpoints/*.pt` 的 `backbone.0.weight` 形状不匹配，无法 `load_state_dict`。需要重新训练或重新做行为克隆。

## 环境要求

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

# 行为克隆热启动（推荐，先让策略有基本玩法再 RL 微调）
python train/generate_data.py --games 200 --grid-max 8 --out checkpoints/dataset_8x8.npz
python train/pretrain.py --data checkpoints/dataset_8x8.npz --out checkpoints/pretrained_8x8.pt --epochs 6

# 正式训练（配置在 config.py，默认开启地图尺寸课程 + self-play；可从预训练权重起步）
python train/train.py
python train/train.py --init-from-pretrained checkpoints/pretrained_8x8.pt

# 注：warm-start 微调会自动把 entropy 系数降到 0.001、LR 降到 1e-4（默认 0.01/3e-4 会毁掉
# 已预训练的确定性策略，实测熵会从 2.6 涨到 6、胜率归零）。可用 --ent-coef/--lr 覆盖。

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
# 对内置 bot 的胜率评估（--grid-min/max 可指定地图尺寸，如对齐真实 1v1 的 18-20）
python train/eval.py --model checkpoints/latest.pt --grid-min 18 --grid-max 20 --games 30 --greedy

# 加入 human_exe（vendored EklipZ bot）作为评估对手（每局较慢）
python train/eval.py --model checkpoints/latest.pt --opponents random,expander,human_exe \
    --grid-min 18 --grid-max 20 --games 10

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

- **奖励**：非终止步使用势能差塑形（`potential(s') - potential(s)`），势能为**相对优势代理**：比值势能（兵力比/地盘比）+ **相对 log 势能** + **净城堡数**（我方-对方），终止步给 +1/-1。相对 log 势能中**对手衰减项权重是自身增长项的 2 倍**（`ABS_OPP_WEIGHT=0.2` vs `ABS_WEIGHT=0.1`）——大图上无限扩张进中立地会一直涨自身项、导致"不进攻拖局"，调高对手衰减让**削减对手比扩张更划算**，驱动主动进攻
- **时间相关项**（鼓励攻击效率、减少拖局）：平局（截断）给 `-DRAW_PENALTY=0.7`（比输 -1 小）；每步时间惩罚超过 `STEP_PENALTY_START=100` 步后开始累计，且**每 `STEP_PENALTY_INTERVAL=200` 步递增 `STEP_PENALTY_INCREMENT=0.002`**——拖得越久越贵，强迫终结而不是龟缩。`GAMMA=0.995`（非 0.99）让胜负信号能传到 600 步前的早期决策
- **兵力增长（默认对齐真实 generals.io）**：本环境 2 步 = 真实 1 turn，`ARMY_INCREMENT_RATE=50`（每格 +1 每 25 turn）+ `CITY_GROWTH_PERIOD=2`（城堡/将军每 turn +1）。对局 ~150-500 步完结，节奏真实。`agents/game_patch.py` 的 `FastGame` 从 config 读这两个值，想加速训练可临时设 25/1，但快增长训出的策略不迁移真实游戏
- **记忆增强观测（31 通道，论文 arXiv:2507.06825）**：15 通道原始观测 → 31 通道。`agents/memory_aug.py` 的 `MemoryAugmenter` 在**网络内部**按批维护记忆 buffer（纯 dict，不进 `state_dict`）：① `seen`（已探索格子的 3×3 膨胀累积，避免重复搜索）；② `enemy_seen`（对手视野累积，标记敌方领土与已侦察区域）；③ 已见过的王城/城堡/山脉（通道 6/7/8，重进迷雾后仍保留）；④ 敌人在每格**上次出现的兵力值 + 距今步数**（通道 21/22）；⑤ **双方近 4 步的兵力变化历史栈**（通道 23–30，即论文的"last moves"表示）。`FastGame` 只保留兵力增长 patch，不再注入过程式记忆（由网络内 augmenter 承担）。`forward` 是纯无状态的前向（31 通道），PPO update / 行为克隆 / 评估直接吃存储的增强观测。自对弈对手的 `PPOAgent` 按 `env_keys` 为每个 env 隔离记忆，共享快照不串扰
- **Self-play**：对手池保留最近 N 个策略快照；self-play 比例随训练**从 0.5 斜坡升到 0.75**（2000 迭代内），前期以 bot 为主学基础，后期以历史自己为对手练对抗，保留少量 bot 防策略漂移
- **行为克隆热启动**：用规则 bot（ExpanderAgent）或 **human_exe**（vendored EklipZ bot）自对弈生成数据集，监督预训练策略再 RL 微调（论文的核心技巧，能显著加速收敛）。生成数据时记录 31 通道增强观测，只保留演示者的合法动作（与论文一致）
  - 用 human_exe 生成 18–20 图的数据集（generals.io 1v1 标准尺寸，bot 按此调优）：
    ```bash
    python train/generate_data.py --demonstrator human_exe --games 200 --max-samples 200000 \
        --grid-min 18 --grid-max 20 --pad-to 24 --out checkpoints/dataset_hx.npz
    python train/pretrain.py --data checkpoints/dataset_hx.npz --out checkpoints/pretrained_hx.pt --epochs 6
    python train/train.py --grid-min 18 --grid-max 20 --no-curriculum \
        --init-from-pretrained checkpoints/pretrained_hx.pt
    ```
  - 注意：human_exe 每步约 0.2–1s（MST / 网络流 / 纯 Python 背包），200 局 18–20 图约 1–2 小时，适合后台跑。小规模验证用 `--games 5 --grid-min 14 --grid-max 14`
- **地图尺寸课程**：默认分阶段 8×8 → 10×12 → 12×14。每阶段对当前地图上 bot 的贪婪评估胜率达到阈值即晋级（或达到该阶段迭代数上限防止卡死），网络按最大地图固定尺寸，跨阶段断点兼容。见 `config.py` 的 `CURRICULUM`
- **动作空间**：每 tick 从 9 通道动作（4 方向 × 全量/半量 + 跳过）中采样，用 `compute_valid_move_mask` 屏蔽非法移动

## 项目结构（新增文件）

```
agents/memory_aug.py       # 网络内记忆增强：15→31 通道观测（论文 2507.06825）
agents/game_patch.py       # FastGame：增长对齐真实 generals.io（monkeypatch 注入 env）
agents/deploy_agent.py     # 部署包装：padding/priority 对齐（记忆由网络内 augmenter 提供）
agents/human_exe_agent.py  # 把 vendored EklipZ bot 桥接到 gymnasium env（HumanExeAgent）
agents/human_exe/          # EklipZgit/generals-bot @ 1e6510d 的 vendored 副本
train/generate_data.py     # 规则 bot / human_exe 自对弈生成行为克隆数据集 (.npz)
train/pretrain.py          # 监督预训练策略（--init-from-pretrained 接入 train.py）
deploy.py                  # 部署到真实 generals.io 服务器（私有房 / 1v1 排位）
verify_human_exe.py        # human_exe 全对局冒烟测试
```

## human_exe（EklipZ bot）迁移

[EklipZgit/generals-bot](https://github.com/EklipZgit/generals-bot)（human server 上叫 Human.exe）是纯算法 generals.io 1v1 bot（MST 集结 / 启发式扩张 / 迷雾信念 / 敌方王城预测）。已 **vendored** 到 `agents/human_exe/`（commit `1e6510d`），`agents/human_exe_agent.py` 把它接进 gymnasium env：从 `Game` 构建全知 `MapBase` → 用 `generate_player_map` 建迷雾视野 + **迷雾正确的逐回合同步**（不用 bot 自带的 `set_map_vision`，它会把隐藏兵力和敌方王城泄漏进迷雾）→ `EklipZBot.find_move()` → 转成 gymnasium action。

用途：
- **行为克隆数据**：`--demonstrator human_exe`（见上）
- **自对弈对手池**：`python train/train.py --use-human-exe-opponent`（慢，按 flag 开启；每个 env 独立实例，`compute_opponent_actions` 走 `act_on_game`）
- **评估对手**：`--opponents human_exe`

需要额外依赖（`pip install logbook ortools disjoint-set unionfind heapq_max networkx`，建议用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）。vendored 副本已打补丁：`pcst_fast` 无 Windows/py3.12 wheel → 用 networkx Steiner 树回退；C++ 背包扩展不构建 → 纯 Python `KnapsackUtilsPy` 回退；加 `from __future__ import annotations`（原 bot 依赖 Python 3.14 的延迟注解）；关闭 bot 调试断言与文件日志。已知局限：ortools 的 MIP 后端（CBC/SCIP）在 Windows 不可用，个别 bot bug 会让单步回退为 pass，但整体对战表现正常（冒烟测试对 Random/Expander 全胜）。

## 部署到真实 generals.io

训练好的策略可部署到真实服务器对战真人。**关键**：模型是在 31 通道记忆增强观测上训练的，而真实游戏只有 15 通道原始观测——`DeployPPOAgent` 靠**网络内的 `MemoryAugmenter`** 在运行时把记忆特征重建出来（只额外做 padding 对齐与 `priority=0`），保证训练/部署输入一致。

```bash
# 需要 generals.io 账号的 user_id（账户设置里的密钥，保密）
# 私有房间（bot 服务器，强制开局，适合先测试）
python deploy.py --user-id <SECRET> --lobby <房间号> --username "[Bot] PPOBot"

# 公开 1v1 排位
python deploy.py --user-id <SECRET> --one-v1 --public --username "[Bot] PPOBot"

# 其它选项：--model 换模型、--device cuda、--num-games N 限局数
```

限制：网络按 24×24 填充输入训练（`PAD_TO=24`），地图须 ≤ 24×24；generals.io 1v1 标准图在范围内。

## 后续路线

- numpy 环境训练收敛慢时可迁移到 generals-bots 的 [JAX 版](https://github.com/strakam/generals-bots)（快 100 倍以上）
