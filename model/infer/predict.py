import json
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path', default=None, type=str)
parser.add_argument('--data_file', default=None, type=str,
                    help="A file that contains instructions (one instruction per line)")
parser.add_argument('--with_prompt', action='store_true', help="wrap the input with the prompt automatically")
parser.add_argument('--interactive', action='store_true', help="run in the instruction mode (single-turn)")
parser.add_argument('--predictions_file', default='./predictions.json', type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
parser.add_argument('--alpha', type=str, default="1.0",
                    help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
parser.add_argument('--torch_dtype', default="float16", type=str, choices=["auto", "bfloat16", "float16", "float32"], 
                    help="Load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.")

args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    min_new_tokens=10,
    max_new_tokens=1536
)

sample_data = ["为什么要减少污染，保护环境？"]

def generate_prompt(input_text):
    return 'Human: ' + input_text + ' </s> Assistant: '

if __name__ == '__main__':
    load_type = (
        args.torch_dtype if args.torch_dtype in ["auto", None]
        else getattr(torch, args.torch_dtype)
        )
    if args.only_cpu:
        device_map = "cpu"
    else:
        device_map = "cuda"
    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, 
        trust_remote_code=True
        resume_download=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=load_type,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    )

    model_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size != tokenzier_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
    else:
        model = base_model

    # test data
    if args.data_file is None:
        examples = sample_data
    else:
        with open(args.data_file, 'r') as f:
            examples = [l.strip() for l in f.readlines()]
        print("first 10 examples:")
        for example in examples[:10]:
            print(example)
    model.eval()

    with torch.no_grad():
        if args.interactive:
            print("Start inference with instruction mode.")

            while True:
                raw_input_text = input("Input:\n")
                if len(raw_input_text.strip()) == 0:
                    break
                if args.with_prompt:
                    input_text = generate_prompt(raw_input_text)
                    print(f'\ninput_text: {input_text}')
                else:
                    input_text = raw_input_text
                inputs = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs['attention_mask'].to(model.device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    generation_config=generation_config
                )

                s = generation_output[0]
                s = s[inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(s, skip_special_tokens=True)
                print('-' * 25)
                print(f"\nResponse:\n{response}")
        else:
            print("Start inference.")
            results = []
            for index, example in enumerate(examples):
                if args.with_prompt is True:
                    input_text = generate_prompt(example)
                else:
                    input_text = example
                inputs = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs['attention_mask'].to(model.device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config=generation_config
                )
                s = generation_output[0]
                s = s[inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(s, skip_special_tokens=True)
                print(f"======={index}=======")
                print(f"Input: {example}\n")
                print(f"Output: {response}\n")

                results.append({"Input": input_text, "Output": response})

            dirname = os.path.dirname(args.predictions_file)
            os.makedirs(dirname, exist_ok=True)
            with open(args.predictions_file, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            generation_config.save_pretrained('./')
