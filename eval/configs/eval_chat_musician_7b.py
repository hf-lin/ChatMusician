from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.partitioners import NaivePartitioner


with read_base():
    from .datasets.collections.base_medium_llama import mmlu_datasets
    from .datasets.music_theory_bench.music_theory_bench_ppl_zero_shot import music_theory_bench_datasets_zero_shot
    from .datasets.music_theory_bench.music_theory_bench_ppl_few_shot import music_theory_bench_datasets_few_shot
    from .models.chat_musician.hf_chat_musician import models

datasets = [
    *mmlu_datasets,
    *music_theory_bench_datasets_zero_shot,
    *music_theory_bench_datasets_few_shot
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=8,
        task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask)
    ),
)
