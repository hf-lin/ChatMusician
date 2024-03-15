datasets = [
    [
        dict(
            abbr='knowledge_zero_shot',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict(
                        A=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='knowledge',
            path='m-a-p/MusicTheoryBench',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='test'),
            type='opencompass.datasets.MusicTheoryBenchDataset'),
    ],
]
models = [
    dict(
        abbr='ChatMusician-Base',
        batch_padding=False,
        batch_size=8,
        max_out_len=100,
        max_seq_len=2048,
        model_kwargs=dict(
            device_map='cuda', from_tf=True, resume_download=True),
        path='m-a-p/ChatMusician-Base',
        run_cfg=dict(num_gpus=1, num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False),
        tokenizer_path='m-a-p/ChatMusician-Base',
        type='opencompass.models.HuggingFaceCausalLM'),
]
work_dir = './outputs/default/20240304_233400'
