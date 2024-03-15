datasets = [
    [
        dict(
            abbr='music_benchmark_music_knowledge_with_reasoning',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='music_knowledge_with_reasoning',
            path='./data/music_benchmark/',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MusicBenchmarkDataset'),
    ],
]
models = [
    dict(
        abbr='musician_Yi_6B_v1_epoch2',
        batch_padding=False,
        batch_size=8,
        max_out_len=100,
        max_seq_len=2048,
        model_kwargs=dict(device_map='auto', trust_remote_code=True),
        path=
        '/data/yrb/Chat-Musician/model/checkpoints/musician_Yi_6B_v1/checkpoints/epoch2',
        run_cfg=dict(num_gpus=1, num_procs=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False),
        tokenizer_path=
        '/data/yrb/Chat-Musician/model/checkpoints/musician_Yi_6B_v1/checkpoints/epoch2',
        type='opencompass.models.HuggingFaceCausalLM'),
]
work_dir = './outputs/default/20231130_000052'
