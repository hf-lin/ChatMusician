from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import MusicTheoryBenchDataset


music_theory_bench_reader_cfg = dict(
    input_columns=["input", "A", "B", "C", "D"],
    output_column="target",
    train_split='dev'
)

music_theory_bench_all_sets = ["knowledge", "reasoning"]

music_theory_bench_datasets_few_shot = []
for _name in music_theory_bench_all_sets:
    _hint = 'Read the following questions from the four options (A, B, C and D) given in each question. Choose the best option.'
    music_theory_bench_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template={
                opt:
                f"{{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: {opt}\n"
                for opt in ["A", "B", "C", "D"]
            },
        ),
        prompt_template=dict(
            type=PromptTemplate,
            template={
                opt:
                f"{_hint}\n</E>{{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: {opt}"
                for opt in ["A", "B", "C", "D"]
            },
            ice_token="</E>",
        ),
        retriever=dict(type=FixKRetriever),
        inferencer=dict(type=PPLInferencer, fix_id_list=[0, 1, 2, 3, 4]),
    )

    music_theory_bench_eval_cfg = dict(evaluator=dict(type=AccEvaluator), )

    music_theory_bench_datasets_few_shot.append(
        dict(
            abbr=f"{_name}_few_shot",
            type=MusicTheoryBenchDataset,
            path="m-a-p/MusicTheoryBench",
            name=_name,
            reader_cfg=music_theory_bench_reader_cfg,
            infer_cfg=music_theory_bench_infer_cfg,
            eval_cfg=music_theory_bench_eval_cfg,
        ))

del _name, _hint
