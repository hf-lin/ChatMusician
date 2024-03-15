from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='Yi-34B',
        path="01-ai/Yi-34B",
        tokenizer_path='01-ai/Yi-34B',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              trust_remote_code=True
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=4, num_procs=1),
    )
]