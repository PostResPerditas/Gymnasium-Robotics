# Task 1:
base: /data/Project/Gymnasium-Robotics 下的项目使用 conda 环境 gymnasium
base: 本机装有 cuda 但是你可能没有权限检测到；涉及训练工作请启用 cuda
前情提要：我们尝试实现抓取状态生成和手内重定位工作，以实现 桌面抓取 -> 手内操作 -> 指定抓取状态的工作
先前的项目中，我们基于 dexonomy 项目完成了抓取状态生成，现在我希望通过 rl 方法实现手内重定位操作，以从初始接触状态转移到目标接触状态

我尝试通过 Gymnasium-Robotics 项目进行尝试；我们首先通过 shadowhand 进行分析。现在我下载了项目开源库，我想要知道如何进行/测试/可视化一个手内重定位的训练过程。

## 结论先行

这个仓库只提供 Gymnasium-Robotics 环境和测试，不自带训练脚本。ShadowHand 的手内操作基线建议按下面的课程推进：

1. `HandManipulateBlockRotateZ-v1`
   - 只绕方块 z 轴旋转，忽略目标位置，最适合先验证训练链路。
2. `HandManipulateBlockRotateParallel-v1`
   - 加入与坐标轴平行的 x/y 姿态目标。
3. `HandManipulateBlockRotateXYZ-v1`
   - 任意三维姿态目标。
4. `HandManipulateBlockFull-v1` 或别名 `HandManipulateBlock-v1`
   - 同时包含目标位置和目标姿态。
5. 接触状态研究时换成 `_ContinuousTouchSensors` 或 `_BooleanTouchSensors` 环境，例如 `HandManipulateBlockRotateZ_ContinuousTouchSensors-v1`。

当前环境里的任务目标是物体 7D 位姿：

- `observation`: 手部关节、速度、物体速度、物体 7D 位姿。
- `achieved_goal`: 当前物体 7D 位姿。
- `desired_goal`: 目标物体 7D 位姿。

所以它能直接做“物体位姿重定位”。如果目标是“从初始接触状态到指定接触状态”，需要后续把接触模式也放进 goal/reward；触觉环境已经把 92 维接触传感器追加到了 observation 末尾，但默认 HER goal 仍然只看物体位姿。

## 已验证状态

截至 2026-06-11，本机 `gymnasium` conda 环境验证结果：

- `HandManipulateBlockRotateZ-v1` 可以 reset、step、`rgb_array` 离屏渲染。
- 渲染帧尺寸是 `(480, 480, 3)`。
- ShadowHand manipulate 测试通过：`6 passed`。
- 当前包版本：`gymnasium==1.1.1`、`gymnasium_robotics==1.4.2`、`mujoco==3.1.6`、`numpy==1.24.4`。
- 当前 conda 环境是 Python 3.8，且缺少 `torch`、`stable_baselines3`、`tensorboard`。

注意：官方当前 PyTorch 和 Stable-Baselines3 都推荐/要求 Python 3.10+；因此正式训练建议新建或升级到 Python 3.10+ 的 RL 环境。PyTorch CUDA 安装命令以官方 selector 为准，Stable-Baselines3 的 HER 文档也明确建议 GoalEnv 使用 `HerReplayBuffer + SAC/DDPG/TD3/DQN`，且 Dict observation 要用 `MultiInputPolicy`。

参考：

- PyTorch install: https://pytorch.org/get-started/locally/
- Stable-Baselines3 install: https://stable-baselines3.readthedocs.io/en/master/guide/install.html
- Stable-Baselines3 HER: https://stable-baselines3.readthedocs.io/en/master/modules/her.html

## 环境测试

```bash
conda run -n gymnasium pytest tests/envs/hand/test_manipulate.py tests/envs/hand/test_manipulate_touch_sensors.py -q
```

使用新增脚本做一次 reset/step/render 冒烟测试：

```bash
conda run -n gymnasium python scripts/shadowhand_train_sac_her.py \
  --env-id HandManipulateBlockRotateZ-v1 \
  --check-env-only
```

随机策略录制一段 gif，用于确认可视化链路：

```bash
conda run -n gymnasium python scripts/shadowhand_play.py \
  --env-id HandManipulateBlockRotateZ-v1 \
  --episodes 1 \
  --video-dir runs/videos/random
```

## 训练依赖

推荐新建 Python 3.10+ 环境，例如：

```bash
conda create -n gymnas-rl python=3.10
conda run -n gymnas-rl pip install -e .
```

按本机 CUDA/driver 选择 PyTorch CUDA wheel。示例，具体 `cuXXX` 以 PyTorch 官网 selector 为准：

```bash
conda run -n gymnas-rl pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121
```

安装训练和可视化依赖：

```bash
conda run -n gymnas-rl pip install "stable-baselines3[extra]"
```

验证 CUDA：

```bash
conda run -n gymnas-rl python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## SB3 训练：SAC/HER、PPO 和并行环境

先从 `RotateZ` 训练，默认脚本会把 SB3 的 device 设为 `cuda`：

```bash
conda run -n gymnas-rl python scripts/shadowhand_train_sac_her.py \
  --env-id HandManipulateBlockRotateZ-v1 \
  --timesteps 1000000 \
  --device cuda \
  --run-name block_rotatez_sac_her

python scripts/shadowhand_train_sac_her.py \
  --env-id HandManipulateBlockRotateZ-v1 \
  --timesteps 4000000 \
  --batch-size 1024 \
  --buffer-size 1000000 \
  --learning-starts 10000 \
  --n-sampled-goal 8 \
  --eval-freq 50000 \
  --save-freq 200000 \
  --device cuda \
  --run-name block_rotatez_sac_her_4m_bs1024_her8
```

现在训练脚本也支持从 `scripts/config` 读取配置，并在迁移到另一台主机时用 `--output-root` 指定完整导出根目录：

```bash
conda run -n gymnas-rl python scripts/shadowhand_train_sac_her.py \
  --config block_rotatez_sac_her_4m_parallel.json \
  --output-root /data/runs/gymnasium-robotics/shadowhand
```

等价地，也可以直接在命令行指定算法和并行环境：

```bash
conda run -n gymnas-rl python scripts/shadowhand_train_sac_her.py \
  --algo sac \
  --use-her \
  --env-id HandManipulateBlockRotateZ-v1 \
  --timesteps 4000000 \
  --batch-size 1024 \
  --buffer-size 1000000 \
  --learning-starts 10000 \
  --n-sampled-goal 8 \
  --num-envs 8 \
  --vec-env subproc \
  --eval-freq 50000 \
  --save-freq 200000 \
  --device cuda \
  --output-root /data/runs/gymnasium-robotics/shadowhand \
  --run-name block_rotatez_sac_her_4m_bs1024_her8_parallel
```

PPO 示例配置：

```bash
python scripts/shadowhand_train_sac_her.py \
  --config block_rotatez_ppo_parallel.json \
  --output-root /data/runs/gymnasium-robotics/shadowhand
```

更适合 PPO 的 shaped-reward 配置已经拆到独立入口，避免影响通用 SAC/HER/PPO 脚本：

```bash
python scripts/shadowhand_train_ppo_isaac_shaped.py \
  --config block_rotatez_ppo_isaac_shaped_parallel.json \
  --output-root /mnt/ssd/Bennkyou/PROJECT/data_PRO/runs/Gymnasium-Robotics/shadowhand
  --output-root /data/runs/gymnasium-robotics/shadowhand
```

这个入口由 `scripts/shadowhand_train_ppo_isaac_shaped.py`、`scripts/shadowhand_wrappers.py` 和 `scripts/config/block_rotatez_ppo_isaac_shaped_parallel.json` 组成。它尽量贴近 IsaacGymEnvs 的 ShadowHand PPO 思路：使用手写 shaped reward、`gamma=0.99`、`learning_rate=5e-4`、`vf_coef=4.0`、ELU 大网络、observation normalization 和更大的并行采样数。注意 Gymnasium-Robotics 这里仍然是 MuJoCo CPU 物理仿真，不能像 IsaacGym 一样使用 16384 个 GPU 物理环境；`num_envs=32` 是多进程 CPU 采样起点，可以按机器 CPU 核数覆盖，例如：

```bash
python scripts/shadowhand_train_ppo_isaac_shaped.py \
  --config block_rotatez_ppo_isaac_shaped_parallel.json \
  --num-envs 64 \
  --output-root /mnt/ssd/Bennkyou/PROJECT/data_PRO/runs/Gymnasium-Robotics/shadowhand
```

说明：

- `--algo` 可选 `sac`、`td3`、`ddpg`、`ppo`。
- `sac`、`td3`、`ddpg` 支持 `--use-her`；`ppo` 是 on-policy 算法，不能使用 HER replay buffer。
- `--reward-mode isaac` 和 `--vec-normalize` 只在 `scripts/shadowhand_train_ppo_isaac_shaped.py` 这条实验入口中使用；通用 `scripts/shadowhand_train_sac_her.py` 不加载 shaped reward。
- `--reward-mode isaac` 会用 `goal distance + rotation reward + action penalty + success bonus` 替换环境原始 sparse reward。
- `--vec-normalize` 会保存 `vecnormalize.pkl` 到模型目录；实验可视化脚本会自动读取它。
- `--config` 可以传完整路径，也可以只传 `scripts/config` 下的文件名。
- `--output-root` 会导出到 `<root>/logs/<run-name>`、`<root>/models/<run-name>`、`<root>/tensorboard`。
- `--run-dir` 可以指定单次运行的完整目录，默认子目录是 `logs/`、`models/`、`tensorboard/`。
- `--num-envs > 1` 时会启用 SB3 vectorized env；`--vec-env subproc` 使用多进程采样，通常比单环境更适合 MuJoCo 这类 CPU 仿真瓶颈。
- 并行环境下，脚本会把 `eval_freq` 和 `save_freq` 按 `num_envs` 折算，仍按总环境步数理解。

训练产物：

- 日志：`runs/shadowhand/block_rotatez_sac_her/`
- TensorBoard：`runs/tensorboard/`
- 模型：`runs/models/block_rotatez_sac_her/final_model.zip`
- 最优模型：`runs/models/block_rotatez_sac_her/best_model.zip`

查看 TensorBoard：

```bash
conda run -n gymnas-rl tensorboard --logdir runs/tensorboard
```

进阶环境命令只需要替换 `--env-id`：

```bash
--env-id HandManipulateBlockRotateParallel-v1
--env-id HandManipulateBlockRotateXYZ-v1
--env-id HandManipulateBlockFull-v1
--env-id HandManipulateBlockRotateZ_ContinuousTouchSensors-v1
```

## 策略测试和可视化

如果训练命令是：

```bash
python scripts/shadowhand_train_ppo_isaac_shaped.py \
  --config block_rotatez_ppo_isaac_shaped_parallel.json \
  --output-root /mnt/ssd/Bennkyou/PROJECT/data_PRO/runs/Gymnasium-Robotics/shadowhand
```

打开 MuJoCo human viewer 查看最终模型：

```bash
python scripts/shadowhand_play_ppo_isaac_shaped.py \
  --model-path /mnt/ssd/Bennkyou/PROJECT/data_PRO/runs/Gymnasium-Robotics/shadowhand/models/block_rotatez_ppo_isaac_shaped_parallel/final_model.zip \
  --episodes 16 \
  --human \
  --device cuda
```

查看评估最优模型时只需要替换模型路径：

```bash
python scripts/shadowhand_play_ppo_isaac_shaped.py \
  --model-path /mnt/ssd/Bennkyou/PROJECT/data_PRO/runs/Gymnasium-Robotics/shadowhand/models/block_rotatez_ppo_isaac_shaped_parallel/best_model.zip \
  --episodes 16 \
  --human \
  --device cuda
```

`shadowhand_play_ppo_isaac_shaped.py` 会自动读取模型目录下的 `config.json`，识别 `algo=ppo`、`env_id=HandManipulateBlockRotateZ-v1`、`reward_mode=isaac` 和环境参数；如果模型目录有 `vecnormalize.pkl`，也会自动加载 observation normalization 统计量。如果模型目录没有 `config.json`，需要手动加 `--algo ppo --env-id HandManipulateBlockRotateZ-v1`。

离屏录制 gif：

```bash
conda run -n gymnas-rl python scripts/shadowhand_play.py \
  --env-id HandManipulateBlockRotateZ-v1 \
  --model-path runs/models/block_rotatez_sac_her/final_model.zip \
  --episodes 5 \
  --video-dir runs/videos/block_rotatez_sac_her \
  --device cuda
```

打开 MuJoCo human viewer：

```bash
python scripts/shadowhand_play.py \
  --env-id HandManipulateBlockRotateZ-v1 \
  --episodes 16 \
  --human \
  --device cuda \
  --model-path runs/models/block_rotatez_sac_her_4m_bs1024_her8/final_model.zip \
```

## 对接 dexonomy 抓取状态的下一步

Gymnasium-Robotics 现成 ShadowHand 环境的初始状态可以通过 `initial_qpos` 和 `randomize_initial_position=False`、`randomize_initial_rotation=False` 固定下来；目标物体位姿可以通过定制 `_sample_goal()` 或 reset 后设置 `env.unwrapped.goal` 固定。真正的“目标接触状态”建议做成一个新环境：

- 从 dexonomy 读取初始手姿态、物体位姿、目标手/物体接触状态。
- 继承 `MujocoHandBlockTouchSensorsEnv`，让 observation 保留 92 维 touch sensor。
- 把 `achieved_goal` 从 7D 物体位姿扩展为 `物体位姿 + 接触特征`。
- 把 `desired_goal` 扩展为 `目标物体位姿 + 目标接触特征`。
- 改写 `compute_reward()`：`pose_reward + contact_reward + object_stability_penalty`。
- 先在 `RotateZ`/固定目标上调通，再扩展到完整位姿和真实抓取状态库。

gymnasiu-robo 中的 ppo 环境不适用于训练手内重定位任务
