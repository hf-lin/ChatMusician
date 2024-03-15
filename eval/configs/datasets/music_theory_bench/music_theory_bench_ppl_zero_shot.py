from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MusicTheoryBenchDataset


music_theory_bench_reader_cfg = dict(
    input_columns=["input", "A", "B", "C", "D"],
    output_column="target",
    train_split='test'
)

music_theory_bench_all_sets = ["knowledge", "reasoning"]

music_theory_bench_datasets_zero_shot = []
for _name in music_theory_bench_all_sets:
    _hint = 'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.'
    music_theory_bench_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template={
                opt:
                f"{_hint}\n{{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: {opt}"
                for opt in ["A", "B", "C", "D"]
            },
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=PPLInferencer)
    )

    music_theory_bench_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

    music_theory_bench_datasets_zero_shot.append(
        dict(
            abbr=f"{_name}_zero_shot",
            type=MusicTheoryBenchDataset,
            path="m-a-p/MusicTheoryBench",
            name=_name,
            reader_cfg=music_theory_bench_reader_cfg,
            infer_cfg=music_theory_bench_infer_cfg,
            eval_cfg=music_theory_bench_eval_cfg,
        ))

del _name, _hint
