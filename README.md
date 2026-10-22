# SuperMimic on mjlab

参照 [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) 的独立迁移版本。保留 G1 29 自由度动作跟踪、可微世界模型、
BPTT 策略优化、critic、回放缓存和 LAFAN1 数据流程，仿真改用 **mjlab + MuJoCo Warp**。
直接加载原 G1 URDF 转换得到的 MJCF，不需要 Isaac Lab、Isaac Sim、Newton 或 USD。
参考项目基线提交：`a1508d42c9d1699a52e8233ab16d08c488441ab9`。

这是参考实现的后端迁移，不是 SuperMimic 官方代码。参考项目的训练成绩不能直接视为本版本的成绩；
本版本需要重新训练和评估。

## 安装与数据

环境：Linux、Python 3.12、uv、支持 CUDA 12.8 的 NVIDIA 驱动、C/C++ 编译器。
离线视频渲染另外需要系统 `ffmpeg` 和 EGL/OpenGL 库。

```bash
git clone https://github.com/humble02/supermimic_mjlab_vibed.git
cd supermimic_mjlab_vibed
bash tools/setup_local.sh
bash tools/download_data.sh
```

依赖安装到本目录 `.venv/`，完整版本由 `uv.lock` 固定。`run_local.sh` 设置项目内的
Warp、TorchInductor、Matplotlib 缓存，默认 `OMP_NUM_THREADS=8`（可用环境变量覆盖）。

已有完整数据时可以复制导入，无需重新下载：

```bash
bash tools/download_data.sh --source /path/to/supermimic_vibed/data
```

目录结构：

```text
data/
  lafan1_g1/*.csv                         # 40 个原始 30 Hz 动作
  g1_description/g1_29dof_rev_1_0.urdf
  g1_description/meshes/*.STL
  g1_description/g1_29dof.xml             # 本地生成
```

模型和动作数据未打包进代码。下载来源是
[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)。
LAFAN1 动作为 CC BY-NC-ND 4.0，G1 描述为 Unitree BSD-3-Clause，代码保留原项目的 [MIT 许可证](LICENSE)。

### 离线服务器

服务器无需访问 GitHub/PyPI。将源码、`data/`、uv 可执行文件，以及在联网机器上准备的离线依赖包
通过 SSH 传入。离线依赖包包含 `requirements.txt`（精确版本）、`wheels/*.whl`，
以及可选的 `sha256.json`（文件名到 SHA256 的映射）。依赖包须匹配 Linux x86_64 / Python 3.12。

```bash
export PATH="$HOME/.local/bin:$PATH"
CUDA_VISIBLE_DEVICES=0 bash tools/setup_offline.sh .cache/deploy/offline
CUDA_VISIBLE_DEVICES=0 ./run_local.sh -m pytest -q
CUDA_VISIBLE_DEVICES=0 ./run_train.sh lafan1_h100
```

离线安装脚本强制 `UV_OFFLINE=1`。`run_local.sh` 直接使用已安装的环境，不会触发依赖下载。
`CUDA_VISIBLE_DEVICES=0` 将程序限定在一张物理 GPU 上；训练中看到的设备为 `cuda:0`。

## 验证

```bash
./run_local.sh -m pytest -q
./run_local.sh tools/test_env.py --num_envs 64 --steps 100
```

pytest 检查动作边界、缓存环绕、状态读写、局部重置、FK、控制周期、执行器、碰撞掩码、
每环境摩擦/质心随机化、超时与终止状态保存。GPU 测试需要 CUDA 和准备好的 G1 模型，
没有这些条件时会明确跳过；`tools/test_env.py` 则直接检查实际动作库。

短训练检查（包含世界模型、策略、critic 更新和 checkpoint）：

```bash
./run_train.sh smoke --num_envs 32 --buffer_steps 64 \
  --wm_batch 16 --pi_batch 16 --wm_batches 1 \
  --iterations 3 --wm_only_iters 1 --warmup_steps 16 \
  --critic_epochs 1 --critic_minibatches 2 \
  --log_every 1 --ckpt_every 3 --no_compile
```

`--no_compile` 用于快速排查。正常训练默认启用 `torch.compile`，首次编译可能需要数分钟。

## 训练、续训、评估和视频

原项目默认配置（4096 环境、`wm_batch=2048`、`pi_batch=4096`）：

```bash
./run_train.sh lafan1_mjlab
```

RTX 5090 / 32 GB 的起始配置：

```bash
./run_train.sh lafan1_mjlab --num_envs 1024 --wm_batch 1024 --pi_batch 1024 --amp bf16
```

训练在前台运行，可自行放进 tmux。原算法默认 4096 个环境，世界模型预热 500 次后开始策略更新。
`pi_batch` 独立控制想象轨迹的数量；调整仿真环境数量时要同时考虑它的显存占用。
控制频率 50 Hz，物理步长 5 ms，每控制步 4 个子步。

```bash
# 续训使用包含优化器状态的 last.pt，保持网络维度相关参数一致。
./run_train.sh lafan1_mjlab --num_envs 1024 --wm_batch 1024 --pi_batch 1024 \
  --resume logs/lafan1_mjlab/last.pt

# 单 checkpoint 评估，可选记录完整轨迹。
./run_local.sh supermimic/evaluate.py --run_dir logs/lafan1_mjlab \
  --ckpt logs/lafan1_mjlab/ckpt_002000.pt --record results/traj.npz

# 轮询新 checkpoint，写入 eval.csv 和逐动作 JSON。
./run_local.sh supermimic/evaluate.py --run_dir logs/lafan1_mjlab --watch

./run_local.sh tools/plot.py logs/lafan1_mjlab --out results/success_curve.png
./run_local.sh tools/render.py results/traj.npz results/demo.mp4 --seconds 10
```

评估对每个动作从第一帧开始，关闭噪声和随机化，摩擦系数为 1。
`ckpt_*.pt` 保存评估需要的网络；`last.pt` 还保存优化器。续训会重新填充未保存的回放缓存。

## 本机验证（2026-10-22）

在 RTX 5090 / 32 GB、项目自己的 `.venv` 中完成：

- `pytest`：7 项全部通过，包括实际 CUDA 仿真测试。
- 40 个 LAFAN1 动作加载为 441,140 个 50 Hz 帧；64 环境连续推进 100 步通过。
  状态读写最大误差 `2.98e-8`，身体 FK 姿态/速度最大误差 `9.54e-7`。
- eager 与 `torch.compile` 两条训练路径都完成世界模型、策略、critic 更新，checkpoint 参数全部有限。
- `last.pt` 成功续训到第 4 次迭代；全 40 动作评估、末帧轨迹记录、失败分析、绘图和 EGL 视频导出通过。
- 1024 环境、`wm_batch=pi_batch=1024`、默认 H=32 / world-model window=8 / buffer=1000，
  eager 短训练完成 2 次迭代（第 2 次包含策略和 critic 更新）。PyTorch 峰值分配 5.56 GiB，
  缓存保留 6.18 GiB；这些数字不包括 Warp/CUDA 驱动分配，也不代表完整训练峰值。

验证日志保存在 `logs/validation/`，短训练产物在 `logs/smoke*` 和 `logs/scale_smoke/`。
这些 checkpoint 仅用于验证功能；尚未进行完整的收敛训练。

## 实现与迁移差异

| 文件 | 职责 |
|---|---|
| `supermimic/robot.py` | MJCF、G1 顺序、原 BeyondMimic PD 增益/armature/力矩限制/action scale |
| `supermimic/env.py` | mjlab Scene / Entity / Simulation、并行物理、随机化、reset sampling |
| `supermimic/motion.py`, `kinematics.py`, `rotations.py` | 动作重采样、关节/身体运动学、旋转运算 |
| `supermimic/mdp.py` | 奖励、观测、硬终止和可微终止 |
| `supermimic/networks.py`, `buffer.py`, `train.py` | 世界模型、策略、critic 和训练 |
| `supermimic/evaluate.py` | 全动作评估、checkpoint 监听、轨迹导出 |

基于 [mjlab 的 Scene/Entity/Simulation 架构](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)，
环境直接使用这些底层组件，保留 SuperMimic 的 transition 接口，没有替换成 PPO。
固定 mjlab `1a00a622`、MuJoCo Warp `875c4caf`、MuJoCo 3.6.0、Warp 1.12.0、PyTorch 2.9.0 CUDA 12.8。

- 四元数从原 Isaac Lab 3 接口的 xyzw 改为 mjlab 的 wxyz；策略状态格式仍是原来的 wxyz。
- 所有 root 速度为世界坐标下的 link 原点速度；不混用质心速度或 MuJoCo freejoint 的局部角速度。
- 每次 reset 和控制步后显式 forward，保证姿态、关节和速度属于同一时刻。
- 摩擦与 torso 质心字段及其派生常量按环境展开；修改质心后重新计算模型常量。
- 仅保留机器人与地面碰撞，关闭自碰撞；使用 URDF 原始碰撞几何。
- mjlab 的内置 position actuator 使用隐式 PD 和力矩限制；MuJoCo 没有对应原
  `velocity_limit_sim` 的硬关节速度上限，因此保留电机额定速度元数据但不额外裁剪速度。
- 修正原实现的仅含末帧的 reset bin、同 bin 同时结束的写入竞争，以及续训时空回放缓存的问题。

物理后端、接触求解和状态刷新语义的变化意味着原 checkpoint 虽可按相同网络结构加载，
性能需要在 mjlab 中重新验证。
