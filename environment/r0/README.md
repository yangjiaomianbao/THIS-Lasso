# R0 环境使用与重建

当前环境为 Julia 1.9.4、Python 3.11.17；两端最小测试均通过。完整状态与实际执行记录见 [R0 报告](../../reports/R0_report.md)。

## 日常使用

```bash
cd /home/yg/THIS-Lasso
source environment/r0/activate.sh
julia --version
python --version
```

激活脚本使用独立 Conda 环境 `.venv-r0/`，并将 Julia Project 和 depot 指向本项目；只影响当前 shell。VS Code 通过 `julia.sh` 包装器加载同一个 depot。

## 保留文件的作用

| 文件 | 用途 |
| --- | --- |
| `activate.sh` | 当前 shell 的环境激活入口 |
| `julia.sh` | VS Code 使用的 Julia 启动包装器 |
| `requirements.in` | 直接依赖及兼容约束，包含选择说明 |
| `requirements.lock.txt` | 实际安装的全部 28 个 Python 发行包版本 |
| `conda-linux-64.lock.txt` | Python 3.11.17 与原生库的精确 Conda 包地址 |
| `julia/Project.toml`、`julia/Manifest.toml` | Julia 直接依赖与完整版本锁定 |
| `upstream_lock.json` | 两个官方仓库的 commit、许可证哈希及文件哈希 |

Conda 安装的 pip 元数据带有构建机的本地路径；Python 锁定文件已将其规范为实际 `pip==26.2.1`，避免重建依赖不存在的构建目录。所有锁定版本已与当前环境逐项核对。

## 重建 Python 环境

仅在目标环境目录尚不存在时执行，以下不是再次安装当前环境的指令：

```bash
cd /home/yg/THIS-Lasso
conda create --prefix /home/yg/THIS-Lasso/.venv-r0 --file environment/r0/conda-linux-64.lock.txt --yes
.venv-r0/bin/python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --requirement environment/r0/requirements.lock.txt
```

原生锁定文件针对 Linux x86_64。当前机器的本地代理曾不可用，安装命令只在该次进程内移除了 HTTP(S)/ALL_PROXY 的大小写变量；没有修改全局代理设置。重建时应使用实际可用的网络配置。

## 重建 Julia 包环境

保留 Julia 1.9.4 本体后，用 Manifest 恢复依赖即可，无需保留一次性的 `Pkg.add` 安装脚本：

```bash
cd /home/yg/THIS-Lasso
source environment/r0/activate.sh
JULIA_PKG_SERVER=https://mirrors.ustc.edu.cn/julia julia --startup-file=no -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Julia 本体位于 `/home/yg/.local/opt/julia-1.9.4/`。安装压缩包已清理，版本、下载来源和官方 SHA-256 校验结果仍保存在报告及 `setup_history.log` 中。

## 人工验收命令

```bash
cd /home/yg/THIS-Lasso
source environment/r0/activate.sh
julia --startup-file=no scripts/smoke_r0.jl
python scripts/smoke_r0.py
python -m pip check
```

在 VS Code 打开此项目，执行命令面板的 `Julia: Start REPL`，确认 `VERSION` 为 1.9.4，并检查 `Base.active_project()` 指向本项目 `environment/r0/julia/Project.toml`。该界面操作留待人工确认；自动测试成功不代表人工验收已勾选。
