import os
import argparse
import json
import math
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import deepspeed
import datasets
from peft import LoraConfig, get_peft_model
from utils import print_trainable_parameters, print_rank_0, to_device, set_random_seed, save_model
from utils import DataCollator
from model import MODE


local_env = os.environ.copy()
os.environ.update(local_env)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboard import SummaryWriter


def parse_args():
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
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument("--show_loss_step", default=10, type=int, help="")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="")
    parser.add_argument("--save_model_step", default=None, type=int, help="")
    # deepspeed features
    parser.add_argument("--ds_file", type=str, default="ds_zero2.json", help="")
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
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()

    with open(args.ds_file, "r", encoding="utf-8") as fh:
        ds_config = json.load(fh)

    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps

    if args.global_rank <= 0:
        tb_write = SummaryWriter()
    set_random_seed(args.seed)
    torch.distributed.barrier()
    
    # load tokenizer
    tokenizer = MODE[args.mode]["tokenizer"].from_pretrained(
                                                args.model_name_or_path,
                                                use_fast=False,
                                                trust_remote_code=True)

    # load model
    if args.train_type == "lora":
        device_map = {'': int(os.environ.get('LOCAL_RANK', '0'))}
        model = MODE[args.mode]["model"].from_pretrained(
                                            args.model_name_or_path,
                                            torch_dtype=torch.float16,
                                            trust_remote_code=True,
                                            device_map=device_map)
        if tokenizer.pad_token_id is None:
            num_new_tokens = tokenizer.add_special_tokens(dict(pad_token="<pad>"))
            embedding_size = model.get_input_embeddings().weight.shape[0]
            print(f'num_new_tokens:{num_new_tokens}')

            if len(tokenizer) != embedding_size:
                model.resize_token_embeddings(len(tokenizer))

        lora_module_name = args.lora_module_name.split(",")
        config = LoraConfig(r=args.lora_dim,
                            lora_alpha=args.lora_alpha,
                            target_modules=lora_module_name,
                            lora_dropout=args.lora_dropout,
                            bias="none",
                            task_type="CAUSAL_LM",
                            inference_mode=False,
                            )
        model = get_peft_model(model, config)
    elif args.train_type == "freeze":
        model = MODE[args.mode]["model"].from_pretrained(args.model_name_or_path)
        freeze_module_name = args.freeze_module_name.split(",")
        for name, param in model.named_parameters():
            if not any(nd in name for nd in freeze_module_name):
                param.requires_grad = False
    elif args.train_type == "ptuning":
        if "glm" not in args.mode:
            raise Exception("only GLM models support P-tuning")
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
        raise Exception("train_type无效")
    if tokenizer.pad_token_id is None:
        num_new_tokens = tokenizer.add_special_tokens(dict(pad_token="<pad>"))
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print(f'num_new_tokens:{num_new_tokens}')
        print(f'len(tokenizer):{len(tokenizer)}')
        if len(tokenizer) != embedding_size:
            model.resize_token_embeddings(len(tokenizer))
    print_rank_0("tokenizer.pad_token: {}".format(tokenizer.pad_token), args.global_rank)
    print_rank_0("tokenizer.eos_token: {}".format(tokenizer.eos_token), args.global_rank)
    
    # load data
    train_dataset = datasets.load_from_disk(args.train_path, keep_in_memory=False)['train']
    
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    data_collator = DataCollator(tokenizer)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size, num_workers=200)

    print_rank_0("len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
    print_rank_0("len(train_dataset) = {}".format(len(train_dataset)), args.global_rank)

    # load optimizer
    ds_config["optimizer"]["params"]["lr"] = args.learning_rate
    ds_config["optimizer"]["params"]["betas"] = (0.9, 0.95)
    ds_config["optimizer"]["params"]["eps"] = 1e-8
    ds_config["optimizer"]["params"]["weight_decay"] = 0.1
    num_training_steps = args.num_train_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print_rank_0("num_training_steps = {}".format(num_training_steps), args.global_rank)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    print_rank_0("num_warmup_steps = {}".format(num_warmup_steps), args.global_rank)
    ds_config["scheduler"]["params"]["total_num_steps"] = num_training_steps
    ds_config["scheduler"]["params"]["warmup_num_steps"] = num_warmup_steps
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["warmup_min_lr"] = args.learning_rate * 0.1

    # print parameters
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print_rank_0(name, 0)
    print_trainable_parameters(model)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # init deepspeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, args=args, config=ds_config,
                                                             dist_init_required=True)
    model.train()
    tr_loss, logging_loss, min_loss = 0.0, 0.0, 0.0
    global_step = 0
    
    # trains
    for epoch in range(args.num_train_epochs):
        print_rank_0("Beginning of Epoch {}/{}, Total Micro Batches {}".format(epoch + 1, args.num_train_epochs,
                                                                               len(train_dataloader)), args.global_rank)
        model.train()

        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch"):
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            tr_loss += loss.item()
            model.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model.step()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                # write loss
                if global_step % args.show_loss_step == 0:
                    print_rank_0("Epoch: {}, step: {}, global_step:{}, loss: {}".format(epoch, step + 1, global_step,
                                                                                        (tr_loss - logging_loss) /
                                                                                        (args.show_loss_step * args.gradient_accumulation_steps)
                                                                                        ),
                                 args.global_rank)
                    print_rank_0("step: {}-{}-{}".format(step + 1, global_step, model.global_steps), args.global_rank)
                    if args.global_rank <= 0:
                        tb_write.add_scalar("train_loss", (tr_loss - logging_loss) /
                                            (args.show_loss_step * args.gradient_accumulation_steps), global_step)
                        logging_loss = tr_loss
                # save model
                if args.save_model_step is not None and global_step % args.save_model_step == 0:
                    if ds_config["zero_optimization"]["stage"] == 3:
                        state_dict = model._zero3_consolidated_16bit_state_dict()
                        if args.global_rank <= 0:
                            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}",
                                       state_dict)
                    else:
                        if args.global_rank <= 0:
                            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")
                    model.train()

        if ds_config["zero_optimization"]["stage"] == 3:
            state_dict = model._zero3_consolidated_16bit_state_dict()
            if args.global_rank <= 0:
                save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}", state_dict)
        else:
            if args.global_rank <= 0:
                save_model(model, tokenizer, args.output_dir, f"epoch-{epoch + 1}-step-{global_step}")


if __name__ == "__main__":
    main()