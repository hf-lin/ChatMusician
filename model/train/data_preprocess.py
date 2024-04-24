import os
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
import argparse
import warnings

# Clean up the output a bit; suppress FutureWarning from datasets.table module
warnings.filterwarnings("ignore", message="promote has been superseded by mode='default'.*", category=FutureWarning)

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=False, trust_remote_code=True)

    def tokenize_function_pt(sample):
        result = tokenizer(sample[args.text_field], return_attention_mask=False, add_special_tokens=False)
        return {'input_ids': result['input_ids'], 'labels': result['input_ids']}

    def tokenize_function_sft(sample):
        max_len = 2048
        max_src_len = 1536

        # Check for None in either theh instruction or input - happens in csv data sets
        #src_tokens = tokenizer.tokenize(sample["instruction"] + sample["input"])
        src_tokens = tokenizer.tokenize("" if sample["instruction"] is None else sample["instruction"] + "" if sample["input"] is None else sample["input"])

        if len(src_tokens) > max_src_len:
            src_tokens = src_tokens[:max_src_len]

        max_tgt_len = max_len - 3 - len(src_tokens)
        tgt_tokens = tokenizer.tokenize(sample["output"])

        if len(tgt_tokens) > max_tgt_len:
            tgt_tokens = tgt_tokens[:max_tgt_len]

        if not sample["output"].strip().endswith("</s>"):
            tokens = src_tokens + tgt_tokens + ["</s>"]
        else:
            tokens = src_tokens + tgt_tokens
        assert len(tokens) <= max_len
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        context_length = len(src_tokens)
        labels = [-100] * context_length + input_ids[context_length:]

        assert len(input_ids) == len(labels)
        if len(input_ids) > max_len:
            input_ids = input_ids[-2048:]
            labels = labels[-2048:]

        return {'input_ids': input_ids, 'labels': labels}

    filename = '.'.join(args.input_file.split("/")[-1].split(".")[:-1]) if os.path.exists(args.input_file) else 'processed_tokens'
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, filename)

    tmp_cache_dir = os.path.join(args.output_dir, filename+"_text")
    os.makedirs(tmp_cache_dir, exist_ok=True)
    print("tmp_cache_dir: ", tmp_cache_dir)

    # raw_dataset = load_dataset(args.input_file, cache_dir=tmp_cache_dir, keep_in_memory=False, encoding="utf8")
    # PG add code to be able to load from a cslf constructed csv file
    source_csv = False
    if args.input_file.endswith(".csv"):
        raw_dataset = load_dataset('csv', data_files=args.input_file, cache_dir=tmp_cache_dir, keep_in_memory=False)
        source_csv = True
    else:
        raw_dataset = load_dataset(args.input_file, cache_dir=tmp_cache_dir, keep_in_memory=False)

    print("remove_column_names:", raw_dataset.column_names['train'])

    if args.tokenize_fn == "pt":
        tokenize_function = tokenize_function_pt
    else:
        tokenize_function = tokenize_function_sft

    if source_csv is True:
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=True,
            keep_in_memory=False,
            desc="Running tokenizer on dataset",
        )
    else:
        tokenized_dataset = raw_dataset.map(
            tokenize_function,
            num_proc=args.preprocessing_num_workers,
            remove_columns=raw_dataset.column_names['train'],
            load_from_cache_file=True,
            keep_in_memory=False,
            cache_file_names = {k: os.path.join(tmp_cache_dir, 'tokenized.arrow') for k in raw_dataset},
            desc="Running tokenizer on dataset",
        )
    if args.filter_by_length is not None:
        tokenized_dataset["train"] = tokenized_dataset["train"].filter(
            lambda x: len(x["input_ids"]) <= args.filter_by_length
        )

    tokenized_dataset.save_to_disk(cache_dir)
    print("tokenized dataset saved to: ", cache_dir)
    print("preprocess done")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokenizer_path', default="m-a-p/ChatMusician-Base", type=str, required=True)
    parser.add_argument('-w', '--preprocessing_num_workers', default=64, type=int)
    parser.add_argument('-i', '--input_file', default="m-a-p/MusicPile-sft", type=str,help="")
    parser.add_argument('-o', '--output_dir', default=None, type=str, help="")
    parser.add_argument('--text_field', default='text', type=str, help="")
    parser.add_argument('--tokenize_fn', default='pt', type=str, choices=["pt", "sft"], help="")
    parser.add_argument('--filter_by_length', default=None, type=int, help="")

    args = parser.parse_args()

    main(args)
