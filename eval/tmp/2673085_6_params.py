models = [
    dict(
        type='opencompass.models.HuggingFaceCausalLM',
        abbr='ChatMusician-Base',
        path='m-a-p/ChatMusician-Base',
        tokenizer_path='m-a-p/ChatMusician-Base',
        tokenizer_kwargs=dict(trust_remote_code=True),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False,
        run_cfg=dict(num_gpus=1, num_procs=1)),
]
datasets = [
    [
        dict(
            abbr='knowledge_few_shot',
            type='opencompass.datasets.MusicTheoryBenchDataset',
            path='m-a-p/MusicTheoryBench',
            name='knowledge',
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
            infer_cfg=dict(
                ice_template=dict(
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate',
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    )),
                prompt_template=dict(
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate',
                    template=dict(
                        A=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    ice_token='</E>'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.FixKRetriever'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer',
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ])),
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator'))),
    ],
]
work_dir = './outputs/default/20240308_213702'
