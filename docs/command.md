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

## SAC + HER 训练

先从 `RotateZ` 训练，默认脚本会把 SB3 的 device 设为 `cuda`：

```bash
conda run -n gymnas-rl python scripts/shadowhand_train_sac_her.py \
  --env-id HandManipulateBlockRotateZ-v1 \
  --timesteps 1000000 \
  --device cuda \
  --run-name block_rotatez_sac_her
```

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
conda run -n gymnas-rl python scripts/shadowhand_play.py \
  --env-id HandManipulateBlockRotateZ-v1 \
  --model-path runs/models/block_rotatez_sac_her/final_model.zip \
  --episodes 3 \
  --human \
  --device cuda
```

## 对接 dexonomy 抓取状态的下一步

Gymnasium-Robotics 现成 ShadowHand 环境的初始状态可以通过 `initial_qpos` 和 `randomize_initial_position=False`、`randomize_initial_rotation=False` 固定下来；目标物体位姿可以通过定制 `_sample_goal()` 或 reset 后设置 `env.unwrapped.goal` 固定。真正的“目标接触状态”建议做成一个新环境：

- 从 dexonomy 读取初始手姿态、物体位姿、目标手/物体接触状态。
- 继承 `MujocoHandBlockTouchSensorsEnv`，让 observation 保留 92 维 touch sensor。
- 把 `achieved_goal` 从 7D 物体位姿扩展为 `物体位姿 + 接触特征`。
- 把 `desired_goal` 扩展为 `目标物体位姿 + 目标接触特征`。
- 改写 `compute_reward()`：`pose_reward + contact_reward + object_stability_penalty`。
- 先在 `RotateZ`/固定目标上调通，再扩展到完整位姿和真实抓取状态库。
