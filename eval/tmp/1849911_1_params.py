datasets = [
    [
        dict(
            abbr='siqa',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict({
                        '1':
                        dict(round=[
                            dict(
                                prompt=
                                '{context}\nQuestion: {question}\nA. {answerA}\nB. {answerB}\nC. {answerC}',
                                role='HUMAN'),
                            dict(prompt='Answer: A', role='BOT'),
                        ]),
                        '2':
                        dict(round=[
                            dict(
                                prompt=
                                '{context}\nQuestion: {question}\nA. {answerA}\nB. {answerB}\nC. {answerC}',
                                role='HUMAN'),
                            dict(prompt='Answer: B', role='BOT'),
                        ]),
                        '3':
                        dict(round=[
                            dict(
                                prompt=
                                '{context}\nQuestion: {question}\nA. {answerA}\nB. {answerB}\nC. {answerC}',
                                role='HUMAN'),
                            dict(prompt='Answer: C', role='BOT'),
                        ])
                    }),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='social_i_qa',
            reader_cfg=dict(
                input_columns=[
                    'context',
                    'question',
                    'answerA',
                    'answerB',
                    'answerC',
                ],
                output_column='label',
                test_split='validation'),
            type='opencompass.datasets.HFDataset'),
    ],
]
models = [
    dict(
        abbr='llama-2-7b',
        batch_size=16,
        max_out_len=100,
        max_seq_len=2048,
        path='./models/llama2/llama/llama-2-7b/',
        run_cfg=dict(num_gpus=1, num_procs=1),
        tokenizer_path='./models/llama2/llama/tokenizer.model',
        type='opencompass.models.Llama2'),
]
work_dir = './outputs/default/20231019_140431'
