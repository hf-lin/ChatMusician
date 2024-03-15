from opencompass.models import HuggingFaceCausalLM

model_path_mapping = {
    "ChatMusician": "m-a-p/ChatMusician",
    "ChatMusician-Base": "m-a-p/ChatMusician-Base"
}

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr=model_abbr,
        path=model_path,
        tokenizer_path=model_path,
        tokenizer_kwargs=dict(
            trust_remote_code=True,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
    for model_abbr, model_path in model_path_mapping.items()
]
