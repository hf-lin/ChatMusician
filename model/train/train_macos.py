import os

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

if IN_COLAB:
  from google.colab import drive
  colab_root = '/content/drive'

  if not os.path.exists(colab_root):
    drive.mount(colab_root)

# Set up huggingface environment

if IN_COLAB:
  os.environ["HF_HOME"] = "/content/drive/MyDrive/cache/huggingface"
  os.environ["HF_DATASETS_CACHE"] = "/content/drive/MyDrive/cache/huggingface/datasets"

elif os.path.exists("/Users/petergreis/Dropbox/Leeds/Project"):
  os.environ["HF_HOME"] = "/Users/petergreis/Dropbox/Leeds/Project/huggingface"
  os.environ["HF_DATASETS_CACHE"] = "/Users/petergreis/Dropbox/Leeds/Project/huggingface/cache"

import argparse
import json
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import datasets
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from utils import print_trainable_parameters, print_rank_0, to_device, set_random_seed, save_model
from utils import DataCollator
from model import MODE
import resource
import platform
import wandb

# Setup optimizer and scheduler
# Replace DeepSpeed optimizer and scheduler setup with PyTorch equivalents
# For simplicity, assuming an optimizer like AdamW and a linear warmup scheduler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, Trainer, TrainingArguments

local_env = os.environ.copy()
os.environ.update(local_env)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter

def parse_args():
    # Existing argument parsing remains the same, remove deepspeed specific args
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model_name_or_path", type=str, help="", required=True)
    # DataSet
    parser.add_argument("--train_path", default="", type=str, help="")
    parser.add_argument("--max_len", type=int, default=1024, help="")
    parser.add_argument("--max_src_len", type=int, default=256, help="")
    parser.add_argument("--is_skip", action='store_true', help="")
    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--output_dir", type=str, default=None, help="")
    parser.add_argument("--mode", type=str, default="llama", help="")
    parser.add_argument("--train_type", type=str, default="lora", help="")
    parser.add_argument("--seed", type=int, default=1234, help="")

    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    parser.add_argument("--save_model_step", default=None, type=int, help="")
    # LoRA
    parser.add_argument("--lora_dim", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=30, help="")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="")
    parser.add_argument("--lora_module_name", type=str, default="query_key_value", help="")
    # Freeze
    parser.add_argument("--freeze_module_name", type=str, default="layers.27.", help="")
    # P-tuning
    parser.add_argument('--pre_seq_len', type=int, default=16, help='')
    parser.add_argument('--prefix_projection', type=bool, default=True, help='')
    return parser.parse_args()

def main():

    cpu_only = False
    # Log everything to wandb
    os.environ["WANDB_WATCH"]= "all"

    # Fix the open file limit of 256 on macOS
    if platform.system() == 'Darwin':
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Set the new soft and hard limits
        new_limit = 65536  # You can adjust this value according to your needs
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))
        # Verify the new limits
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)

    args = parse_args()
    wandb.init(config=args)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        args.global_rank = 0
    elif torch.cuda.is_available():
       device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # device = torch.device("cpu")
    cpu_ony = True if device == torch.device("cpu") else False

    # Set random seed
    set_random_seed(args.seed)

    # Load tokenizer and model as before
    print("args.mode: ", args.mode)
    print("model_name_or_path: ", args.model_name_or_path)
    print("train_type: ", args.train_type)

    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)

    # load model
    if args.train_type == "lora":
        print(">>> device_map set to: ", device)
        #model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True, device_map=device)
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path, torch_dtype=torch.float32, device_map=device)

        if tokenizer.pad_token_id is None:
            num_new_tokens = tokenizer.add_special_tokens(dict(pad_token="<pad>"))
            embedding_size = model.get_input_embeddings().weight.shape[0]
            print(f'num_new_tokens:{num_new_tokens}')

            if len(tokenizer) != embedding_size:
                model.resize_token_embeddings(len(tokenizer))

        lora_module_name = args.lora_module_name.split(",")

        # Referred to as peft_params in other articles
        config = LoraConfig(r=args.lora_dim,
                            lora_alpha=args.lora_alpha,
                            target_modules=lora_module_name,
                            lora_dropout=args.lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            inference_mode=False)
        model = get_peft_model(model, config)

    elif args.train_type == "freeze":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
        freeze_module_name = args.freeze_module_name.split(",")
        for name, param in model.named_parameters():
            if not any(nd in name for nd in freeze_module_name):
                param.requires_grad = False
    elif args.train_type == "ptuning":
        if "glm" not in args.mode:
            raise Exception("0 GLM models support P-tuning")
        config = MODE[args.mode]["config"].from_pretrained(args.model_name_or_path)
        config.pre_seq_len = args.pre_seq_len
        config.prefix_projection = args.prefix_projection
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path, config=config)
        for name, param in model.named_parameters():
            if not any(nd in name for nd in ["prefix_encoder"]):
                param.requires_grad = False
    elif args.train_type == "all":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
    else:
        raise Exception("train_type")
    if tokenizer.pad_token_id is None:
        num_new_tokens = tokenizer.add_special_tokens(dict(pad_token="<pad>"))
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print(f'num_new_tokens:{num_new_tokens}')
        print(f'len(tokenizer):{len(tokenizer)}')
        if len(tokenizer) != embedding_size:
            model.resize_token_embeddings(len(tokenizer))

    print_rank_0("tokenizer.pad_token: {}".format(tokenizer.pad_token), args.global_rank)
    print_rank_0("tokenizer.eos_token: {}".format(tokenizer.eos_token), args.global_rank)

    # Load data
    train_dataset = datasets.load_from_disk(args.train_path, keep_in_memory=False)['train']
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_training_steps = args.num_train_epochs * len(train_dataset) // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_ratio*num_training_steps, num_training_steps=num_training_steps)

# in part modified with info from: https://www.datacamp.com/tutorial/fine-tuning-llama-2
    training_params = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        # per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim='adamw_torch',
        save_steps=25,
        logging_steps=25,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=False,
        #fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type="linear",
        report_to='wandb',
        use_cpu=cpu_only
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=training_params,
        optimizers=(optimizer, scheduler)
    )

    trainer.train()

    # Final model save
    save_dir = args.output_dir

    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(">>> Saved model and tokenizer to: ", save_dir)

if __name__ == "__main__":
    main()
