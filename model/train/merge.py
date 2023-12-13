import torch
import argparse
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_model_dir', default=None, type=str, help='')
    parser.add_argument('--model_dir', default=None, type=str, help='')
    parser.add_argument('--output_dir', default=None, type=str, help='')

    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    base_model = LlamaForCausalLM.from_pretrained(args.ori_model_dir, torch_dtype=torch.float16)
    lora_model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    lora_model.to("cpu")
    model = lora_model.merge_and_unload()
    LlamaForCausalLM.save_pretrained(model, args.output_dir, max_shard_size="2GB")
    if args.output_dir and args.output_dir!=args.model_dir:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.output_dir)