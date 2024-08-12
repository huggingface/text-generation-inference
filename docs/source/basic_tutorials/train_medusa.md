# Train Medusa

This tutorial will show you how to train a Medusa model on a dataset of your choice. Please check out the [speculation documentation](../conceptual/speculation.md) for more information on how Medusa works and speculation in general.

## What are the benefits of training a Medusa model?

Training Medusa heads can greatly improve the speed of generation. Medusa adds extra "heads" to LLMs to predict multiple future tokens simultaneously. When augmenting a model with Medusa, the original model stays untouched, and only the new heads are fine-tuned during training.

One of the most important things is to have a good dataset (with similar data to what will be used in production) because Medusa has a much higher hit-rate when the generation is in-domain.

If you train Medusa on a dataset that is very different from the one you will use in production then the model will not be able to predict the future tokens accurately and consequently the speedup will be minimal or non-existent.

## Self-distillation (Generating data for training)

There are many methods for preparing data for training, but one of the easiest and most effective ways is to "self-distill" the data. This means that you can use the same model to generate the data that you will use to train the model.

Essentially, you prompt the model with a similar input to what you will use in production and the model will generate the output.

We'll use this output to help train the medusa heads to predict the `n+1`, `n+2`, `n+3`, etc tokens in the sequence.

## Training

The original implementation of Medusa is available at [https://github.com/FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) and we'll follow a very similar process to train the model as described on the original repository.

### Getting Started

There are two methods for training the model:

- `torchrun` that is a wrapper around `torch.distributed.launch`
- a forked version of `axlotl` that supports Medusa

In this tutorial we'll use `torchrun` to train the model as it is the most straightforward way to train the model but similar steps can be followed to train the model using `axlotl` if you prefer.

### Training with `torchrun`

```bash
mkdir medusa-training
cd medusa-training

pyenv install 3.10
pyenv local 3.10

uv venv -p 3.10
source .venv/bin/activate
```

Now lets clone the original `Medusa` repository and install the library.

```bash
git clone https://github.com/FasterDecoding/Medusa.git
cd Medusa
pip install -e .
```

Next we'll need some data to train on, we can use the `ShareGPT_Vicuna_unfiltered` dataset that is available on the Hugging Face Hub.

```bash
apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
```

Currently our directory structure looks like this:

```bash
.
├── assets
├── CITATION.cff
├── create_data.py
├── data_generation
├── deepspeed.json
├── last_run_prepared
├── LICENSE
├── llm_judge
├── medusa
├── medusa_llm.egg-info
├── mistral.json
├── notebooks
├── pyproject.toml
├── README.md
├── ROADMAP.md
├── scripts
├── ShareGPT_Vicuna_unfiltered
│   ├── README.md
│   ├── ShareGPT_2023.05.04v0_Wasteland_Edition.json
│   └── ShareGPT_V4.3_unfiltered_cleaned_split.json
├── simple_gradio_interface.py
├── tiny-llama.json
└── vicuna_7b_qlora_stage1
```

## Start Training

Now the lets generate the data and start training the model. This process will take a while since we are generating data from the model.

First make sure you have an instance of TGI running with the model you want to use for self-distillation.

```bash
model=HuggingFaceH4/zephyr-7b-beta
volume=/home/ubuntu/.cache/huggingface/hub/

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest --model-id $model
```

Now we can generate the data using the `create_data.py` script.

```bash
python create_data.py \
    --input-filename ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --output-filename zephyr_self_distill.json
```

At this point our terminal should look like this:

<div class="flex justify-center">
    <img
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/medusa-train-large.gif"
        width="550"
    />
</div>

> Note: In the screen shot above we are only using a the first 500 examples from the dataset to speed up the process, you should have a much larger dataset for training.

Now we can finally get to the fun part and start training the model!

Using `torchrun` we can easily launch the `medusa` training script with the `zephyr_self_distill.json` configuration file.

> NOTE: If you just self-distilled you may still have the model running, make sure to stop it before starting the training in order to allow all of the resources to be used for training.

```bash
WANDB_MODE=offline torchrun --nproc_per_node=4 medusa/train/train_legacy.py \
    --model_name_or_path HuggingFaceH4/zephyr-7b-beta \
    --data_path zephyr_self_distill.json \
    --bf16 True \
    --output_dir zephyr_out \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 3 \
    --medusa_num_layers 1 \
    --deepspeed deepspeed.json
```

<div class="flex justify-center">
    <img
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/medusa-train-heads-large.gif"
        width="550"
    />
</div>

If successful, you should see the similar output to the one below:

```bash
wandb: Run history:
wandb:                    train/epoch ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇███
wandb:              train/global_step ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇███
wandb:            train/learning_rate ▅███▇▇▆▅▅▄▃▂▂▁▁▁
wandb:                     train/loss ██▆▄▄▃▃▂▂▃▁▁▂▁▁▁
wandb:             train/medusa0_loss ▆▆▇▆▆▅▄▅▃▃▃▃▂▂▂▂▂▃▂▂▂▁▁▁▂▁▁▁▁▁█▁▁▁▂▁▁▁▁▁
wandb:             train/medusa0_top1 ▁▁▁▁▁▁▁▁▃▂▃▃▄▄▄▃▄▃▄▄▅▅▆▅▆▆▇▅▇▇▄▇█▇▅▇█▆▇▇
wandb:             train/medusa1_loss ▇▇█▇▇▆▅▅▃▄▃▃▃▃▃▃▃▃▃▃▂▁▂▂▂▁▁▂▁▁▇▁▁▁▂▁▁▁▁▁
wandb:             train/medusa1_top1 ▁▁▁▁▁▁▁▁▃▂▃▃▃▄▄▃▃▂▃▃▅▅▆▄█▆▇▅▇▇▅█▇▇▅▇█▆▆▇
wandb:             train/medusa2_loss ▃▃▄▄▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁█▁▁▁▂▁▁▁▁▁
wandb:             train/medusa2_top1 ▁▁▁▂▁▁▁▁▂▂▃▃▃▄▄▃▃▂▃▃▅▆▅▄█▆▆▅▆▆▄█▇▇▄▇█▆▆▇
wandb:               train/total_flos ▁
wandb:               train/train_loss ▁
wandb:            train/train_runtime ▁
wandb: train/train_samples_per_second ▁
wandb:   train/train_steps_per_second ▁
wandb:
wandb: Run summary:
wandb:                    train/epoch 2.0
wandb:              train/global_step 16
wandb:            train/learning_rate 0.0
wandb:                     train/loss 14.8906
wandb:             train/medusa0_loss 4.25
wandb:             train/medusa0_top1 0.28809
wandb:             train/medusa1_loss 4.8125
wandb:             train/medusa1_top1 0.22727
wandb:             train/medusa2_loss 5.5
wandb:             train/medusa2_top1 0.17293
wandb:               train/total_flos 0.0
wandb:               train/train_loss 23.98242
wandb:            train/train_runtime 396.9266
wandb: train/train_samples_per_second 2.519
wandb:   train/train_steps_per_second 0.04
```

Last but most importantly, don't forget to push this model to the Hugging Face Hub so you can use it in your projects.

```bash
python -m medusa.hf_utils \
    --folder zephyr_out_medusa_mlp_zephyr-7b-beta_medusa_3_lr_0.001_layers_1 \
    --repo drbh/zephyr_medusa_demo
```

Woo, we've successfully trained a Medusa model and pushed it to the Hugging Face Hub! 🎉
