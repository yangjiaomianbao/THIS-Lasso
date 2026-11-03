# 独立局部采样与连续轨迹秩亏诊断 v1

从仓库根目录运行。本目录是独立版本，审计与预先固定协议见 [AUDIT.md](AUDIT.md)。原 data.py / library.py / screeners.py / backend.py / neural.py、checkpoint 和历史结果不修改。

## API

```python
from experiments.large_scale_screening_v1.common import load_config
from experiments.large_scale_screening_v1.sampling_diagnostics_v1.sampling import generate, coordinates

cfg = load_config()
data = generate(30, 2026110310, 2026110410, cfg['data'],
                sampling_mode='local_iid', delta=0.6)
x, center = coordinates(data)  # 固定网络中心，两种模式一致
trajectory = generate(30, 2026110310, 2026110410, cfg['data'],
                      sampling_mode='trajectory', train_trajectories=64, duration=0.5)
```

默认1024/256/256行；local_iid在固定随机非零中心周围做独立均匀制备，直接计算解析rhs，三套样本使用独立随机数流。group是sample ID，time=NaN，不接入时序神经网络。trajectory直接调用原RK45实现，允许轨迹数/每轨迹点数和时长控制。噪声继续使用原 noisy_derivative，不改定义。

## 命令与执行顺序

使用已有含torch（仅运行历史测试和环境版本记录需要）、NumPy、SciPy、sklearn、matplotlib的环境，不安装/升级依赖。本机现成环境为 `/home/yg/miniconda3/envs/nd3/bin/python`。

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/home/yg/miniconda3/envs/nd3/bin/python
EXP=experiments.large_scale_screening_v1.sampling_diagnostics_v1.runner
OUT=results/large_scale_screening_v1/sampling_diagnostics_v1
"$PYTHON" -B -m unittest discover -s experiments/large_scale_screening_v1/sampling_diagnostics_v1/tests -v
"$PYTHON" -B -m "$EXP" pilot --out "$OUT" --workers 8
"$PYTHON" -B -m "$EXP" formal --out "$OUT" --workers 10
"$PYTHON" -B -m "$EXP" b1 --out "$OUT" --workers 4
"$PYTHON" -B -m "$EXP" b2 --out "$OUT" --workers 4
"$PYTHON" -B -m "$EXP" b3 --out "$OUT" --workers 4
# 本轮在64轨迹已可解但结构F1很低后追加，属于明确标记的探索性激励边界检查：
"$PYTHON" -B -m "$EXP" extension --out "$OUT" --workers 4
"$PYTHON" -B -m "$EXP" matrix --out "$OUT"
"$PYTHON" -B -m "$EXP" report --out "$OUT"
"$PYTHON" -B -m "$EXP" verify --out "$OUT"
```

换新目录复现用 `--out`。相同命令可恢复：核对冻结源码/配置及每个完成任务文件哈希后跳过已完成任务。未完成任务会重算该任务，旧的已完成科研证据不会混入新参数运行。formal有明确的两种子成功率门槛；不会自动放宽BIC、换求解器或启动神经训练。

30种子包含前两种子，不能合并为32种子。每个拓扑内部所有方法、预算和噪声都共享相同状态及单位噪声流。B1/B2/B3只用两个配对网络、5%噪声、K16、Pearson/SCC-OLS/FWL，避免多因素笛卡尔积。

数据和拟合都只输出到新目录。原trajectory CLI的历史输出仍由其冻结协议保护；新子目录属于新增源码，旧CLI的整目录源码哈希检测可能因此要求新输出目录，不能强行绕过其恢复校验。

## 审查结果

结果入口为输出目录的 `REPORT.md`。主表包含筛选与全网结构指标、FP/FN、原BIC成功率和耗时；`target_trials.csv`提供每个目标的数值秩/条件数，`arrays.npz`保留完整奇异值和原生索引。B4采用固定候选以控制库选择因素，原始/标准尺度、中心、阈值敏感性分别记录。解析完整Taylor误差、候选遗漏、回归误差互不替代。

`edge_temporal_PLV_mean`是每条真实边在每条轨迹内的相位差锁定值；短时间缓慢演化也可使它接近1，必须结合频率差、相位差和T控制解读。local_iid不计算时间PLV或轨迹lag-1。

所有结构阈值沿用1e-8。数值满秩、BIC接受、结构质量是三件事；完整网络中任一目标拒绝时，结构指标保留NA，不能用部分目标或零系数填充。
