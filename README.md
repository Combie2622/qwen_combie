# Qwen3.5 + CMExam：阿里云 DSW 单卡实验手册

本方案对应你给出的 DSW 环境：torcheasyrec 镜像、Python 3.11、单张 GPU、8 核 CPU、30 GiB 主机内存。GPU 显存型号未提供，所以任何训练前都先查看 `nvidia-smi` 和 `00_env_check.py` 的报告；代码默认 BF16、SDPA、micro-batch=1、梯度检查点、LoRA、梯度累积和教师 CPU offload。

## 交付目录

- `config.yaml`：DSW 路径、固定 Prompt、全部共同超参数、KD 参数。
- `requirements.txt`：SWIFT/Qwen3.5 环境依赖。
- `scripts/00_env_check.py`：GPU、CUDA、Python、关键包和显存预检。
- `scripts/01_prepare_cmexam.py`：清洗、规范选项、去重、防泄漏、训练长度筛选与数据导出。
- `scripts/02_template_check.py`：比较 4B、0.8B 的 Prompt 序列化。
- `scripts/03_baseline.py`：greedy 推理，支持基座或 LoRA adapter。
- `scripts/04_train_sft.py`：4B / 0.8B LoRA SFT。
- `scripts/05_train_kd.py`：0.8B 学生的离线 GKD + SFT 双 Loss，T=2 / T=4。
- `scripts/06_merge_adapter.py`：将 LoRA 与基座合并的可选工具。
- `scripts/07_score.py`：答案完全匹配准确率、格式率和解析盲评表。
- `scripts/08_plot.py`：准确率柱状图、训练/验证 Loss 曲线。
- `scripts/09_length_audit.py`：训练序列长度审计。
- `scripts/run_*.sh`：DSW Bash 运行入口。

## 先确认实验比较的含义

官方 `Qwen3.5-4B` 是经过后训练的模型检查点，不是 `Qwen3.5-4B-Base`。本方案按你指定的 4B 权重执行，报告写作“未做本项目微调的 4B 初始模型”。0.8B 按你指定的 Base 检查点。若论文必须比较两个 Base，应在正式实验前把 4B 路径换成 `Qwen3.5-4B-Base`，并重新完成所有基线与训练。

主对照关系：

| 组别 | 起始权重 | 训练目标 | 用途 |
|---|---|---|---|
| 4B 初始 | Qwen3.5-4B | 无 | 模型能力基线 |
| 0.8B 初始 | Qwen3.5-0.8B-Base | 无 | 学生基线 |
| 4B-SFT | 同一 4B 检查点 | 硬标签 SFT | 教师模型与微调对照 |
| 0.8B-SFT | 同一 0.8B Base | 硬标签 SFT | 学生普通微调对照 |
| 0.8B-KD+SFT-T2 | 同一 0.8B Base | 前向 KL + 硬标签 SFT，T=2 | 蒸馏温度对照 |
| 0.8B-KD+SFT-T4 | 同一 0.8B Base | 前向 KL + 硬标签 SFT，T=4 | 蒸馏温度对照 |

唯一变量的结论要分层表述：`0.8B-SFT` 对 `0.8B-KD+SFT` 是“加入软标签 KD loss”的比较；T=2 与 T=4 是温度比较。4B 与 0.8B、初始与微调不是单变量比较，不能用来声称差异只来自蒸馏。

## 1. 部署模型、数据和代码

### 1.1 创建持久化目录

在 DSW 文件浏览器终端执行。建议把模型、数据、缓存和输出放到持久化云盘或 OSS/NAS 挂载路径；实例本地临时盘重启后可能消失。

```bash
mkdir -p /mnt/workspace/models /mnt/workspace/cache/modelscope /mnt/workspace/cache/huggingface
mkdir -p /mnt/workspace/qwen35_cmexam/outputs /mnt/workspace/qwen35_cmexam/data
export MODELSCOPE_CACHE=/mnt/workspace/cache/modelscope
export HF_HOME=/mnt/workspace/cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 1.2 复制代码、上传数据

将本包中的 `qwen35_cmexam_dsw/` 整个目录上传到 DSW，例如：`/mnt/workspace/qwen35_cmexam_dsw`。上传 CMExam 官方 CSV 至：

```text
/mnt/workspace/CMExam-main/data/train.csv
/mnt/workspace/CMExam-main/data/val.csv
/mnt/workspace/CMExam-main/data/test_with_annotations.csv
```

数据原始目录只读。Windows 本地 `C:`、`D:` 路径不会自动映射到 DSW；确认文件确实在 Linux 终端显示的挂载点内。

### 1.3 下载 Qwen3.5 检查点

在 DSW 终端设置 ModelScope 缓存后下载：

```bash
modelscope download --model Qwen/Qwen3.5-4B --local_dir /mnt/workspace/models/Qwen3.5-4B
modelscope download --model Qwen/Qwen3.5-0.8B-Base --local_dir /mnt/workspace/models/Qwen3.5-0.8B-Base
```

检查目录内至少有 `config.json`、tokenizer 文件和模型权重分片。若模型文件已从本地上传，跳过下载并修改 `config.yaml` 中路径。修改所有路径后再运行后续步骤。

### 1.4 核对硬件并装包

```bash
cd /mnt/workspace/qwen35_cmexam_dsw
python scripts/00_env_check.py --config config.yaml
nvidia-smi
python -m pip install -r requirements.txt
python -m pip check
swift sft --help
swift rlhf --help
```

环境验收关注：只有一个可见 CUDA GPU；CUDA 可用；PyTorch 不低于 2.5；Transformers 在 `>=5.9,<5.17`；`swift rlhf --help` 包含 `gkd`、`temperature`、`sft_alpha`、`lmbda`、`offload_teacher_model`。不要盲目覆盖镜像自带的 CUDA 版 PyTorch。若依赖解析准备卸载/替换 Torch，停止安装并先确认 wheel 与镜像 CUDA 组合。

**显存决策：**建议完整跑 4B LoRA 与 KD 至少使用 24 GiB GPU。16–24 GiB 可先用本方案的 micro-batch=1、BF16、SDPA、梯度检查点及教师 offload，再用少量样本做显存试跑。低于 16 GiB 风险较高；若需要降低 `training.max_length`，必须在任何正式训练前改配置、重跑数据准备，并让所有训练组都使用新值。不能只给一个实验组降低长度。

## 2. 清理、格式化、划分数据并封锁测试集

CMExam 官方切分是 `train.csv`、`val.csv`、`test_with_annotations.csv`。脚本按题干和选项组成规范化去重键，优先保留 `test > val > train`；同一切分内相同题目若答案冲突则整组排除。无效题、跨切分重复、答案冲突和超长样本都写入 `data/split_audit.csv`。

运行：

```bash
python scripts/02_template_check.py --config config.yaml
python scripts/01_prepare_cmexam.py --config config.yaml
python scripts/09_length_audit.py --config config.yaml
```

产物：

| 文件 | 内容 | 用途 |
|---|---|---|
| `data/train.jsonl` | 题干、选项、答案、解析；过滤掉与 val/test 重复题与超长样本 | 唯一训练数据 |
| `data/val.jsonl` | 清洗后验证集 messages | 训练中选 checkpoint |
| `data/test_inputs.jsonl` | 题干与选项，无答案/解析 | 最终预测输入 |
| `data/test_labels.jsonl` | ID、标准答案、参考解析 | 仅 `07_score.py` 最终评分读取 |
| `data/split_audit.csv` | 每条无效、冲突、重复和长度剔除记录 | 人工核查 |
| `data/dataset_summary.json` | 原始/保留数量、去重、长度剔除、指纹 | 报告复现记录 |

训练序列长度由 `training.max_length=512` 限定，答案+解析最多 256 tokens。预处理用 0.8B tokenizer 按同一对话模板筛掉超限 train/val 记录，所以 SFT 和 GKD 使用同一份样本，训练时都不需要截断。若 length audit 显示大量过滤，请在实验开始前把长度上限统一调大或调整显存方案，然后重跑全套。

测试题只用于去重审计和最后评估。脚本将测试输入、测试标签写入不同文件；`04_train_sft.py`、`05_train_kd.py` 的参数只指向 train 和 val。不要把 `test_labels.jsonl` 交给训练、提示词调整或 checkpoint 选择。

## 3. 冻结 Prompt、数据、超参数和随机种子

唯一 Prompt 在 `config.yaml: prompt.system`。所有基线与训练后模型使用相同系统提示、题目格式、Qwen3.5 模板、`enable_thinking=false` 和 greedy decoding。运行 `02_template_check.py`，确认两 tokenizer 渲染的原始 prompt 字符串完全相同。若失败，先解决模板差异，禁止直接继续。

训练共同配置：

| 参数 | 值 |
|---|---:|
| Seed / data seed | 42 |
| dtype / attention | BF16 / SDPA |
| 最大训练序列 | 512 tokens |
| 每卡 micro-batch | 1 |
| 梯度累积 | 8（有效 batch=8） |
| Epoch | 1 |
| 学习率 / scheduler | 1e-4 / cosine |
| warmup / weight decay / grad norm | 0.03 / 0.01 / 1.0 |
| LoRA | r=8, alpha=16, dropout=0.05, `all-linear` |
| gradient checkpointing | 开 |
| optimizer / scheduler / checkpoint rule | 所有组相同；验证集 `eval_loss` 选最佳模型 |
| 训练采样 | SWIFT seed=42、data_seed=42、相同 train/val JSONL |

4B-SFT、0.8B-SFT、T=2、T=4 从同一份配置和数据生成命令。KD 两次运行只有 `--temperature` 的值不同。第一次正式训练开始后，不再改 Prompt、数据清洗结果、长度、种子、LoRA 或超参数。

## 4. 未训练基线推理

```bash
bash scripts/run_baseline.sh
```

代码分别加载指定 4B 与 0.8B 基座，只读取 `test_inputs.jsonl`，greedy 解码并增量保存：

```text
outputs/4b_initial/predictions.jsonl
outputs/08b_initial/predictions.jsonl
```

推理参数：`do_sample=false`、`num_beams=1`、输入最多 512 tokens、新生成最多 256 tokens。Base 模型即使不按“答案/解析”格式回答，也如实保留；答案格式失败按错误计入准确率，不能用约束解码帮某组修复。

## 5. 4B 和 0.8B 的 LoRA SFT

依次执行：

```bash
bash scripts/run_sft.sh 4b
bash scripts/run_sft.sh 08b
```

每组使用 `train.jsonl` 和 `val.jsonl`；训练只更新 LoRA。检查输出目录中的 `best_adapter_path.txt`，该文件指向用验证集 `eval_loss` 选出的最佳 adapter。验证集仅用于模型选择，测试集不参与训练、早停或选 checkpoint。

## 6. KD + SFT 双 Loss 蒸馏

先确认 `outputs/4b_sft/best_adapter_path.txt` 存在。蒸馏教师由“4B 原始检查点 + 4B-SFT adapter”构成；教师参数冻结且 no-grad，`offload_teacher_model=true` 将教师卸载到 CPU 以释放显存。学生是原始 `Qwen3.5-0.8B-Base`，仅训练它的 LoRA。

```bash
bash scripts/run_kd.sh 2
bash scripts/run_kd.sh 4
```

GKD 配置：

- `lmbda=0`：off-policy，只使用训练集给出的正确答案序列，不用学生在线采样。
- `beta=0`：Forward KL。
- `sft_alpha=1.0`：总损失为 `KD soft loss + 1.0 × hard-label SFT loss`。
- `temperature=2` 或 `4`：两组间唯一配置差异。
- 教师只读取 train 样本的训练前缀和答案序列；val 只计算验证指标；test 不进入蒸馏。
- `logits_topk=0` 表示默认 full-vocabulary 蒸馏。若显存实测不足，可在正式 T=2 前把它统一改成 64，然后 T=2/T=4 都保持 64，并在报告说明是 top-k 近似。不要一组 full vocab、另一组 top-k。

每组产物包含 `command.json`、`experiment_config.json`、checkpoint、`trainer_state.json` 和 `best_adapter_path.txt`。训练日志保存在各自输出目录。

## 7. 六组模型最终评估、解析盲评和图表

在 4B-SFT、0.8B-SFT、T=2、T=4 训练完成后，运行：

```bash
bash scripts/run_all_predictions.sh
bash scripts/run_evaluation.sh
```

脚本会对所有六组模型以同一 test ID 集合重新生成预测。已完成的基线预测会自动跳过。核心指标是 **答案集合完全匹配准确率**：先提取 `答案：` 后面的 A–F，再排序、去重后和参考答案 exact match；多选题必须全选中且不多选才算正确。同步报告答案格式合格率。

解析质量通过盲评，不把 BLEU/ROUGE 当医学正确性。`outputs/explanation_blind_review.csv` 对相同固定抽样、所有模型随机打乱行并匿名化，提供标准答案与参考解析。建议两位评阅者按 0–2 给三项评分：正确性、完整性、无无依据内容；其中无依据内容反向计分，合计 0–6。完成表格后执行：

```bash
python scripts/07_score.py --config config.yaml --review outputs/explanation_blind_review.csv
python scripts/08_plot.py --config config.yaml
```

图表和汇总保存于 `outputs/accuracy_comparison.png`、`outputs/loss_curves.png`、`outputs/metrics_summary.json`。报告中写明 test 只在全部设置冻结后用于最终评估；盲评结果用于比较解释质量。

## 运行顺序一览

```text
环境预检与安装
  → 下载模型、挂载数据、修改 config.yaml 路径
  → Prompt 模板检查
  → 清洗/去重/长度筛选，审计 split_audit.csv
  → 保存 4B/0.8B 初始基线
  → 4B SFT
  → 0.8B SFT
  → 0.8B KD+SFT T=2
  → 0.8B KD+SFT T=4
  → 六模型统一测试推理
  → 评分、盲评、绘图、归档环境和配置
```

## 实验记录与复现归档

保存模型版本/文件 hash、CMExam 原始与清洗后样本数、去重规则、Prompt、模板检查结果、seed、GPU 型号/显存、Torch/CUDA/Transformers/SWIFT 版本、每组命令、训练用时、峰值显存、最佳 adapter 路径、预测结果、指标和盲评表。不要把数据集原始文件或模型权重误打包上传到公开仓库。

## 官方资料

- [Qwen3.5-4B 模型卡](https://www.modelscope.cn/models/Qwen/Qwen3.5-4B)
- [Qwen3.5-0.8B-Base 模型卡](https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B-Base)
- [ModelScope SWIFT Qwen3.5 实践](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_8-Best-Practice.md)
- [SWIFT 蒸馏 GKD 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Distillation.md)
- [CMExam 官方数据集仓库](https://github.com/williamliujl/CMExam)
