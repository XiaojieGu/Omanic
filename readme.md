
<div align="center">
<h1><a href="https://arxiv.org/pdf/2603.16654" style="color:#68edcb">Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models</a></h1>

[![arXiv](https://img.shields.io/badge/arXiv-2603.16654-b31b1b.svg?style=plastic)](https://arxiv.org/pdf/2603.16654)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange?style=plastic)](https://huggingface.co/datasets/li-lab/Omanic)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=plastic)](https://github.com/XiaojieGu/Omanic)
</div>


## Environment Setup

### SFT (`omanic_sft`)

```bash
bash environment/setup_omanic_sft_env.sh
conda activate omanic_sft
```

### RL (`omanic_rl`)


```bash
bash environment/setup_omanic_rl_env.sh
conda activate omanic_rl
```

## Data prepare


Download the raw dataset files `OmanicSynth.jsonl` and `OmanicBench.jsonl`.

```bash
python data/download_omanic.py
```

For SFT, convert the raw dataset files into `OmanicSynth_sft.json` and `OmanicBench_sft.json`.

```bash
python data/covert_to_sft.py
```

For RL, convert the raw dataset files into `OmanicSynth_rl.json` and `OmanicBench_rl.json`.
The converted RL data uses `data_source="omanic"`, which routes reward computation to `verl/utils/reward_score/omanic.py`.

```bash
python data/convert_to_rl.py
```

## Training

### SFT


```bash
conda activate omanic_sft
cd LlamaFactory
```

For `Llama-3.3-70B`
```bash
nohup bash -c '
module load cuda/12.2.2
export CUDA_HOME=$CUDA_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omanic_sft
export FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=4
llamafactory-cli train examples/train_lora/llama70B_omanic.yaml
' > train_llama70B_omanic.log 2> train_llama70B_omanic.err < /dev/null &
```

For `Qwen3-8B`
```bash
nohup bash -c '
module load cuda/12.2.2
export CUDA_HOME=$CUDA_PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate omanic_sft
export FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=4
llamafactory-cli train examples/train_full/qwen3_8B_oamnic.yaml
' > train_qwen3_8B_oamnic.log 2> train_qwen3_8B_oamnic.err < /dev/null &
```

### RL


```bash
conda activate omanic_rl
cd verl
module load cuda/12.2.2
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```



```bash
nohup bash examples/grpo_trainer/run_qwen3-8b_omanic.sh > grpo_train.log 2>&1 < /dev/null &
```

## Eval

### For GPU-based evaluation

### Llama-3.3-70B

LoRA merge is handled in memory during evaluation.

```bash
python eval/local_eval.py \
  --base-model meta-llama/Llama-3.3-70B-Instruct \
  --mode direct \
  --lora-path LlamaFactory/saves/llama3.3-70b/lora \
  --input data/OmanicBench.jsonl \
  --batch-size 256
```

Use `--mode cot` if you want chain-of-thought style evaluation.

### Qwen/Qwen3-8B

This evaluates the full fine-tuned model directly without LoRA merge.

```bash
python eval/local_eval.py \
  --model-path LlamaFactory/saves/qwen3-8b/full \
  --mode direct \
  --input data/OmanicBench.jsonl \
  --batch-size 256
```

### For API-based evaluation

Set your OpenRouter API key in the shell before running the script:

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

The default input file is `data/OmanicBench.jsonl`, and results will be written to `eval/results`.

Specify a single model with `--model`. You can use either the full OpenRouter model ID or a supported alias.

Examples:

```bash
python eval/open_eval.py \
  --model openai/gpt-5.4 \
  --mode direct

python eval/open_eval.py \
  --model anthropic/claude-sonnet-4.6 \
  --mode cot
```

Use `--model all` to evaluate every model listed in `eval/open_eval.py`:

```bash
python eval/open_eval.py \
  --model all \
  --mode direct
```

You can also override the default input path when needed:

```bash
python eval/open_eval.py \
  --model GPT-4o \
  --mode direct \
  --input data/OmanicBench.jsonl
```


## Contact

For any inquiries, please reach out at **peettherapynoys@gmail.com**



## Citation

If you find Omanic useful for your research and applications, please cite:

```bibtex
@article{gu2026omanic,
  title={Omanic: Towards Step-wise Evaluation of Multi-hop Reasoning in Large Language Models},
  author={Gu, Xiaojie and Tong, Sherry T and Feng, Aosong and Han, Sophia Simeng and Lu, Jinghui and Chen, Yingjian and Iwasawa, Yusuke and Matsuo, Yutaka and Park, Chanjun and Ying, Rex and Li, Irene},
  journal={arXiv preprint arXiv:2603.16654},
  year={2026}
}
```
