from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import HFDataset

siqa_reader_cfg = dict(
    input_columns=['context', 'question', 'answerA', 'answerB', 'answerC'],
    output_column='label',
    test_split='validation')

siqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={
            1: 'The following makes sense:\n {context} \nQ: {question}\nA: {answerA}',
            2: 'The following makes sense:\n {context} \nQ: {question}\nA: {answerB}',
            3: 'The following makes sense:\n {context} \nQ: {question}\nA: {answerC}',
        }),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=PPLInferencer))

siqa_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

siqa_datasets = [
    dict(
        abbr="siqa",
        type=HFDataset,
        path='social_i_qa',
        reader_cfg=siqa_reader_cfg,
        infer_cfg=siqa_infer_cfg,
        eval_cfg=siqa_eval_cfg)
]
