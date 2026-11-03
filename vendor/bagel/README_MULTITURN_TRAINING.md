# BAGEL 多轮交错图文训练复现说明

本文档只说明本仓库新增并实际使用的 `multiturn_jsonl` 训练模式。它的目标是把纯文本问答、图片理解问答、图片生成、图片编辑、多轮交错图文对话统一转换成与 BAGEL 默认推理格式一致的训练样本。

下面所有路径都用占位符表示，使用时替换成你机器上的真实绝对路径：

```text
<BAGEL_REPO>       BAGEL 仓库根目录，例如 /path/to/BAGEL
<BASE_MODEL_DIR>  官方 BAGEL base 权重目录，例如 <BAGEL_REPO>/models/BAGEL-7B-MoT
<DATA_JSONL>      你的训练数据 JSON 或 JSONL 文件
<IMAGE_ROOT>      你的图片根目录，建议用绝对路径
<OUTPUT_ROOT>     训练输出目录，例如 <BAGEL_REPO>/train_lwt/outputs
```

## 1. 仓库内容

核心文件：

```text
<BAGEL_REPO>/
  data/multiturn_jsonl_dataset.py
  data/dataset_info.py
  data/configs/*.yaml
  train/pretrain_unified_navit.py
  train_lwt/run_multiturn_lora_template.sh
  train_lwt/run_vlm_understand_55k_lora.sh
  train_lwt/run_unify_weave_uie_select_recall_lora.sh
  tools/export_ckpt_to_official_layout_hardcoded.py
  environment-bagel.yml
  models/README.md
```

Git 只保存代码、配置、脚本、文档。模型权重、训练数据、转换后的大 JSONL、日志、WandB 缓存、checkpoint、导出的 official layout 都不应该提交。

`models/README.md` 是占位说明。真正权重需要手动放到 `<BASE_MODEL_DIR>`。

## 2. 环境配置

推荐直接使用本仓库导出的环境：

```bash
cd <BAGEL_REPO>
conda env create -f environment-bagel.yml
conda activate bagel
export PYTHONPATH=<BAGEL_REPO>:${PYTHONPATH:-}
```

如果手动安装，核心依赖包括 Python 3.10、PyTorch CUDA 版、transformers、accelerate、safetensors、peft、wandb、Pillow、numpy、flash-attn 等。当前服务器环境中 PyTorch 是 `2.5.1+cu124`。

## 3. Base 模型准备

把官方 BAGEL base 权重放到 `<BASE_MODEL_DIR>`，例如：

```text
<BAGEL_REPO>/models/BAGEL-7B-MoT/
```

目录中应包含官方 layout 所需文件，例如：

```text
ema.safetensors
ae.safetensors
config.json
llm_config.json
vit_config.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
preprocessor_config.json
generation_config.json
```

训练时使用：

```bash
--model_path <BASE_MODEL_DIR>
--finetune_from_hf True
--finetune_from_ema True
--resume_model_only True
```

含义是从官方 base/EMA 权重初始化，然后训练本次新增的 LoRA 和解冻模块。

## 4. 准备数据集

最少只需要两样东西：

1. 一个 JSON 或 JSONL 文件：`<DATA_JSONL>`
2. 一个图片根目录绝对路径：`<IMAGE_ROOT>`

如果 JSON/JSONL 里的图片已经全部是绝对路径，`<IMAGE_ROOT>` 仍然需要在注册表里填写，但不会影响绝对路径解析。如果图片是相对路径，加载器会用：

```text
真实图片路径 = <IMAGE_ROOT> / JSONL 中的相对图片路径
```

建议最终训练文件使用 JSONL：每一行是一个 episode。一个 episode 可以只有 1 轮，也可以有多轮。单轮理解数据集也可以直接使用这个多轮格式，相当于每个 episode 只有一个 turn。

### 4.1 最小 JSONL 格式

每一行结构如下：

```json
{
  "id": "sample-0001",
  "turns": [
    {
      "turn_id": 1,
      "task_type": "vlm_understand",
      "render": {
        "query": {
          "messages": [
            {"role": "user", "type": "image", "content": "Image #1"},
            {"role": "user", "type": "text", "content": "Image #1 中有什么？"}
          ],
          "images": ["0001/source.jpg"]
        },
        "answer": {
          "messages": [
            {"role": "assistant", "type": "text", "content": "图中是一只杯子。"}
          ],
          "images": []
        }
      }
    }
  ]
}
```

`task_type` 支持：

- `vlm_understand`: 图片或文本理解问答，答案是文本。
- `select_mcq`: 选择题理解任务，答案是文本。
- `plan_struct`: 结构化规划或结构化文本输出，答案是文本。
- `t2i`: 文生图，query 是文本，answer 是目标图片。
- `edit`: 图片编辑，query 通常包含源图和编辑指令，answer 是目标图片。
- `interleaved_mm`: 图文交错输出，answer 可以同时包含文本和目标图片。

### 4.2 图片字段怎么写

`messages` 里的 `Image #1`、`Image #2` 是普通文本引用，会原样进入 ChatML。不要把它替换成真实图片，也不要插入特殊 image token。

真实图片路径只放在：

```text
render.query.images
render.answer.images
```

例如图片理解：

```json
"query": {
  "messages": [
    {"role": "user", "type": "image", "content": "Image #1"},
    {"role": "user", "type": "text", "content": "请描述 Image #1。"}
  ],
  "images": ["relative/or/absolute/source.jpg"]
},
"answer": {
  "messages": [
    {"role": "assistant", "type": "text", "content": "这是一张室内照片。"}
  ],
  "images": []
}
```

例如图片编辑：

```json
"query": {
  "messages": [
    {"role": "user", "type": "image", "content": "Image #1"},
    {"role": "user", "type": "text", "content": "把 Image #1 改成夜晚风格。"}
  ],
  "images": ["source.jpg"]
},
"answer": {
  "messages": [
    {"role": "assistant", "type": "image", "content": "Image #2"}
  ],
  "images": ["target.jpg"]
}
```

对 `t2i`、`edit`、`interleaved_mm`，需要被监督生成的目标图必须放在 `render.answer.images`。

## 5. 上下文拼接语义

`data/multiturn_jsonl_dataset.py` 会把每个 episode 展开成逐轮训练样本。

如果一个 episode 有 5 轮，它会产生 5 个训练样本：

```text
样本 1: 第 1 轮 user query -> 第 1 轮 assistant answer
样本 2: 第 1 轮历史 + 第 2 轮 user query -> 第 2 轮 assistant answer
样本 3: 第 1-2 轮历史 + 第 3 轮 user query -> 第 3 轮 assistant answer
...
样本 5: 第 1-4 轮历史 + 第 5 轮 user query -> 第 5 轮 assistant answer
```

拼接目标是对齐 BAGEL 默认推理方式：

1. 当前第 `k` 轮训练样本包含第 1 到第 `k` 轮 user query，以及第 1 到第 `k-1` 轮 assistant answer。
2. 当前第 `k` 轮 assistant answer 不放进上下文，而是作为监督目标接在最后。
3. 每轮 user query 会先插入该轮首次出现的真实图片 visual tokens，然后插入该轮 ChatML 文本。
4. `messages.content` 保持原文，不会把 `Image #N` 替换成图片 token。
5. 历史文本和历史图片 loss 为 0。
6. 当前 answer 文本或目标图片 loss 为 1。

ChatML 标签格式：

```text
<|im_start|>user
用户文本
<|im_end|>
<|im_start|>assistant
历史 assistant 文本
<|im_end|>
<|im_start|>assistant
当前答案从这里开始监督
```

输入图片按真实路径去重。比如第 1 轮出现 `Image #1`，第 3 轮又引用同一张图片并新增 `Image #2`，构造第 3 轮样本时不会重复插入 `Image #1`，只会插入新增的 `Image #2`。构造第 4 轮样本时，历史里仍然保留第 1 轮和第 3 轮各自的文本位置，图片 visual tokens 则按首次出现的位置插入一次。

图片分支：

- 理解轮输入图走 ViT visual tokens。
- 生成/编辑轮的条件图按默认生成推理需要同时提供 VAE 条件和 ViT 条件。
- 生成/编辑的目标图走 VAE target，并对目标图开 loss。

因此这个模式可以训练：

- 纯文本问答
- 图片输入理解问答
- 文生图
- 图片编辑
- 图文交错多轮对话

## 6. 注册数据集

在 `data/dataset_info.py` 的 `DATASET_INFO['multiturn_jsonl']` 下新增一个条目：

```python
'my_multiturn': {
    'data_dir': '<IMAGE_ROOT>',
    'jsonl_path': '<DATA_JSONL>',
    'num_total_samples': 12345,
},
```

`num_total_samples` 填 episode 数，不是展开后的 turn 数。比如 JSONL 有 5108 行，每行 5 轮，则这里填 5108。

然后创建配置文件，例如 `data/configs/my_multiturn.yaml`：

```yaml
multiturn_jsonl:
  dataset_names:
  - my_multiturn
  image_transform_args:
    image_stride: 16
    max_image_size: 1024
    min_image_size: 512
  vit_image_transform_args:
    image_stride: 14
    max_image_size: 768
    min_image_size: 378
    max_pixels: 589_824
  is_mandatory: true
  shuffle_lines: true
  shuffle_seed: 0
  only_current_turn_images: false
  num_used_data:
  - 12345
  weight: 1
```

参数说明：

- `dataset_names`: 必须和 `data/dataset_info.py` 里注册的名字一致。
- `image_transform_args`: VAE/generation 分支图片预处理，影响生成和编辑的图片 token 数。
- `vit_image_transform_args`: 理解分支图片预处理，影响输入图片理解 token 数。
- `shuffle_lines`: 是否按 episode 打乱。
- `only_current_turn_images`: 兼容旧参数。为了保持默认推理对齐，保持 `false`。
- `num_used_data`: 填 episode 数，和注册表保持一致即可。
- `weight`: 多数据集混合训练时的采样权重。

已有两个数据集配置可参考：

```text
data/configs/vlm_understand_55k.yaml
data/configs/unify_weave_uie_select_recall.yaml
```

## 7. 启动训练

通用模板脚本：

```bash
cd <BAGEL_REPO>
bash train_lwt/run_multiturn_lora_template.sh
```

通常只需要改这些变量：

```bash
export REPO_DIR=<BAGEL_REPO>
export TORCHRUN_BIN=<CONDA_ENV_DIR>/bin/torchrun
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MODEL_PATH=<BASE_MODEL_DIR>
export DATASET_CONFIG_FILE=./data/configs/my_multiturn.yaml
export RUN_TAG=bagel_my_multiturn_lora
export WANDB_PROJECT=bagel_my_multiturn
export OUTPUT_PATH=<OUTPUT_ROOT>/bagel_my_multiturn_lora
export TOTAL_STEPS=10000
export SAVE_EVERY=750
bash train_lwt/run_multiturn_lora_template.sh
```

当前建议：除了路径、GPU 卡、`TOTAL_STEPS`、`SAVE_EVERY`、run 名称、WandB 项目名这类和实验组织相关的参数，其它训练参数尽量保持现有脚本默认值。除非你明确知道自己要改变训练语义或资源配置，否则不要随意改 `visual_gen`、`visual_und`、`training_mode`、LoRA、freeze、token budget 等参数。

已有两个可直接运行的脚本：

```bash
bash train_lwt/run_vlm_understand_55k_lora.sh
bash train_lwt/run_unify_weave_uie_select_recall_lora.sh
```

`CUDA_VISIBLE_DEVICES` 控制哪些 GPU 参与训练，`NPROC_PER_NODE`/`--nproc_per_node` 应等于可见 GPU 数。当前训练是 FSDP 数据并行/参数分片，不是流水线并行；每个 rank 都执行完整 forward/backward 的所有层，但参数、梯度、优化器状态按 FSDP shard 分布在不同 GPU 上。

## 8. 训练参数说明

建议保持默认的关键参数：

- `--training_mode default`: 使用默认 BAGEL 训练/推理格式。
- `--visual_und True`: 启用图片理解路径。
- `--visual_gen True`: 启用图片生成/编辑路径。
- `--use_lora True`: 对语言模型线性层挂 LoRA。
- `--lora_r 16 --lora_alpha 32 --lora_dropout 0.05`: 当前使用的 LoRA 配置。
- `--freeze_vit True`: 冻结 ViT。
- `--freeze_vae True`: 冻结 VAE。
- `--freeze_und False`: 不冻结理解/连接相关可训练模块。
- `--disable_ema True`: 不维护 EMA 全量副本，checkpoint 主要保存本次额外训练的权重。
- `--checkpoint_weight_dtype bf16`: checkpoint 权重保存为 bf16。
- `--expected_num_tokens 14336`: packer 期望每个 step 每 rank 处理的 token 预算。
- `--max_num_tokens 16384`: 单个 batch token 上限。
- `--max_num_tokens_per_sample 14336`: 单个展开样本 token 上限，超过会被过滤。
- `--save_every`: 每隔多少 step 保存一次 checkpoint，可以按需要调整。
- `--total_steps`: 总训练 step 数，需要按期望 epoch 数粗略估算。
- `--gen_lr_scale`: 生成相关模块学习率倍率。理解纯文本任务常用 1.0，混合理解/编辑可用当前脚本的 1.6。

## 9. 为什么要估算 epoch

当前训练脚本没有 `--num_epochs` 这类 epoch 参数，只有 `--total_steps`。所以如果想“大概训练 1 个 epoch / 2 个 epoch / 3 个 epoch”，需要先估算：

```text
steps_per_epoch ~= 全部展开样本 token 总数 / (GPU 数 * expected_num_tokens)
```

注意这里的“展开样本”不是 JSONL 行数。多轮 episode 会被展开：

```text
展开样本数 = 所有 episode 的 turn 数总和
```

比如 5108 行 JSONL，每行 5 轮，则展开样本约为：

```text
5108 * 5 = 25540
```

估算步骤：

1. 用真实 loader 或抽样脚本统计展开后每个样本的 `num_tokens`。
2. 关注 mean、p90、p95、p99、max。
3. 确认 `max_num_tokens_per_sample` 能覆盖大多数样本。少量特别长的样本可以被过滤，但不要让大量样本超过上限。
4. 用总 token 数估算 1 epoch 需要多少 step。
5. 按期望 epoch 数设置 `TOTAL_STEPS`。

例如 unify 多轮数据按真实多轮拼接抽样统计，展开样本均值约 4388 tokens，5108 个 episode、约 25540 个展开 turn，8 卡、`expected_num_tokens=14336` 时：

```text
steps_per_epoch ~= 25540 * 4388 / (8 * 14336)
                ~= 976
```

所以：

```text
约 1 epoch: TOTAL_STEPS=1000
约 2 epoch: TOTAL_STEPS=2000
约 3 epoch: TOTAL_STEPS=3000
```

实际 step 会受 packer、过滤超长样本、数据 sharding、图片 token 长度变化影响，所以这是粗略估计，不是严格数学 epoch。

`max_num_tokens_per_sample` 的作用是保护显存和 packer。如果统计发现 p99 是 9000、max 是 14000，那么设置 14336 基本能覆盖绝大多数样本。如果 max 明显超过 14336，需要决定是提高上限，还是接受过滤极长样本。提高上限会增加显存压力，也可能降低 packing 效率。

## 10. Checkpoint 内容

使用当前 LoRA 脚本和 `--disable_ema True` 时，每个 checkpoint 主要包含：

```text
trainable_model.safetensors   # LoRA + 本次解冻训练头/连接模块等可训练权重
optimizer/                    # 继续训练用优化器分片
scheduler.pt                  # 学习率状态
data_status.pt                # 数据读取状态
```

不会每 750 step 保存一份完整 28GB base。用于部署前，需要把 `trainable_model.safetensors` 和 base 合并导出成 official-like layout。

## 11. 导出合并到 base

使用仓库中的导出脚本：

```bash
cd <BAGEL_REPO>
python tools/export_ckpt_to_official_layout_hardcoded.py \
  --ckpt-dir <OUTPUT_ROOT>/<run_name>/checkpoints/0003000 \
  --base-model-dir <BASE_MODEL_DIR> \
  --out-dir <BAGEL_REPO>/train_lwt/official_layout/<run_name>_0003000 \
  --lora-r 16 \
  --lora-alpha 32 \
  --out-dtype bf16
```

导出目录会包含 official-like 文件，例如新的 `ema.safetensors`、tokenizer、config、VAE/Vision 配置等。后续使用时把模型路径指向这个导出目录。

## 12. 数据转换脚本

已有两个转换脚本可作为参考：

```bash
python train_lwt/vlm_understand_55k/convert_to_multiturn.py
python train_lwt/unify_weave_uie_select_recall/convert_unify_paths.py
```

转换时要保证：

- JSONL 每行是完整 episode。
- episode 可以是 1 轮，也可以是多轮。
- 图片路径能由 `<IMAGE_ROOT>` + 相对路径解析，或者直接是绝对路径。
- `Image #1` 等引用文本保留在 `messages.content` 中。
- 真实输入图片路径放到 `render.query.images`。
- 图片生成/编辑目标图路径放到 `render.answer.images`。

## 13. 常见检查

查看日志：

```bash
tail -f <OUTPUT_ROOT>/<run_name>/train.log
```

确认进程：

```bash
cat <OUTPUT_ROOT>/<run_name>/train.pid
ps -fp $(cat <OUTPUT_ROOT>/<run_name>/train.pid)
nvidia-smi
```

如果报 `ModuleNotFoundError` 或本地模块找不到，确认：

```bash
export PYTHONPATH=<BAGEL_REPO>:${PYTHONPATH:-}
```

如果脚本在 Linux 上出现 `$'\r'`，说明文件是 CRLF，需要转成 LF：

```bash
perl -pi -e 's/\r$//' train_lwt/run_*.sh
```

如果报图片不存在，先检查注册的 `<IMAGE_ROOT>` 和 JSONL 中的相对路径是否能拼成真实文件。
