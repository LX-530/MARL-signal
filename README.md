# MARL-signal

## 概览
本仓库主要包含一个基于 QMIX 的多智能体强化学习（MARL）项目，用于动态信号控制与火灾疏散模拟。核心入口在 `QMIX_MARL/main.py`，自定义环境在 `QMIX_MARL/env/environment.py`，CTM 仿真与火灾风险计算在 `QMIX_MARL/utils/CTM/`。

## 目录结构速览
- `QMIX_MARL/main.py`：主入口，组装环境、算法与训练/评估流程。
- `QMIX_MARL/env/`：自定义环境（动态信号与人群/火灾状态）。
- `QMIX_MARL/MARL/`：多智能体算法实现（QMIX/VDN/COMA/QTRAN 等）。
- `QMIX_MARL/utils/CTM/`：CTM 模拟、火灾风险计算、并行采样等工具。
- `QMIX_MARL/fire_info/`：火灾时序数据（CSV）。
- `QMIX_MARL/data_preprocessing/`：数据预处理输出。
- `QMIX_MARL/result/`：训练/评估结果与历史记录。

## 最快验证路径（推荐）
最快的功能验证方式是使用已有的预训练模型进行评估（避免长时间训练）。注意：代码内部使用了相对路径读取 `fire_info/`，所以运行目录必须在 `QMIX_MARL`。

```
cd QMIX_MARL
../.venv/bin/python - <<'PY'
import json
import os
import re
import sys

from env.environment import DynamicSignalEnv
from MARL.runner import Runner
from MARL.common.arguments import (
    get_common_args,
    get_coma_args,
    get_mixer_args,
    get_centralv_args,
    get_reinforce_args,
    get_commnet_args,
    get_g2anet_args,
)

def get_next_folder(base_path, folder_prefix):
    dirs = os.listdir(base_path)
    pattern = re.compile(rf"^{folder_prefix}(\\d+)$")
    max_num = 0
    for name in dirs:
        match = pattern.match(name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return f"{folder_prefix}{max_num + 1}"

sys.argv = [
    "quick_eval",
    "--result_dir",
    "./result",
    "--load_model_dir",
    "./MARL/model",
]
args = get_common_args()

if args.alg.find("coma") > -1:
    args = get_coma_args(args)
elif args.alg.find("central_v") > -1:
    args = get_centralv_args(args)
elif args.alg.find("reinforce") > -1:
    args = get_reinforce_args(args)
else:
    args = get_mixer_args(args)
if args.alg.find("commnet") > -1:
    args = get_commnet_args(args)
if args.alg.find("g2anet") > -1:
    args = get_g2anet_args(args)

args.learn = False
args.load_model = True

env = DynamicSignalEnv(args)
env_info = env.get_env_info()
args.n_actions = env_info["n_actions"]
args.n_agents = env_info["n_agents"]
args.state_shape = env_info["state_shape"]
args.obs_shape = env_info["obs_shape"]
args.episode_limit = env_info["episode_limit"]

save_path = args.result_dir + "/" + args.alg
os.makedirs(save_path, exist_ok=True)
save_path_dir = get_next_folder(save_path, args.map + "_")
train_save_path = save_path + "/" + save_path_dir
train_save_path_model = train_save_path + "/model"
train_save_path_actions = train_save_path + "/historydata/actions_result"
train_save_path_data = train_save_path + "/historydata/train_process"
for path in [train_save_path, train_save_path_model, train_save_path_data, train_save_path_actions]:
    os.makedirs(path, exist_ok=True)
args.save_path = train_save_path

parameters = vars(args)
with open(train_save_path + "/training_parameters.json", "w") as file:
    json.dump(parameters, file, indent=4)

runner = Runner(env, args)
_, reward = runner.evaluate(10101)
print("The ave_reward of {} is  {}".format(args.alg, reward))
PY
```

本次运行输出（示例）：`The ave_reward of qmix is  -40014.02701999999`  
结果写入：`QMIX_MARL/result/qmix/dynamicsignal_1/`（含 `training_parameters.json` 与 `historydata/actions_result/`）。

## 环境创建与安装记录
使用 Python 3.8.10（与 `requirements.txt` 备注一致），在仓库根目录执行：

```
python3 -m venv .venv
.venv/bin/pip install -r QMIX_MARL/requirements.txt
.venv/bin/pip install torch==1.13.1
.venv/bin/pip install sympy==1.12 networkx==3.1
```

说明：
- `requirements.txt` 未包含 `torch`（需按 CUDA/CPU 环境自行安装），也未包含 `sympy` 和 `networkx`（代码依赖但未列出）。
- 运行时会出现 Gym 的弃用警告，不影响本次评估。
