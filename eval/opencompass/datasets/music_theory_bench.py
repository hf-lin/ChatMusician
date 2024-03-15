from datasets import load_dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset

def convert_dataset(dataset_item):
    try:
        if 'abc_score' in dataset_item.keys() and dataset_item['abc_score']:
            stem = dataset_item['stem'].replace('Use the example below to answer the question that follows.', '').strip()
            dataset_item['stem'] = f"{stem}\n{dataset_item['abc_score']}"
        return {
            'input': dataset_item['stem'],
            'A': dataset_item['options']['A'],
            'B': dataset_item['options']['B'],
            'C': dataset_item['options']['C'],
            'D': dataset_item['options']['D'],
            'target': dataset_item['answer']
        }
    except Exception as e:
        raise ValueError(f"missing values detected in {dataset_item}")


@LOAD_DATASET.register_module()
class MusicTheoryBenchDataset(BaseDataset):
    @staticmethod
    def load(path: str, name: str):
        dataset = DatasetDict()
        music_theory_dataset = load_dataset(path)
        dataset['dev'] = music_theory_dataset['dev'].map(convert_dataset)
        dataset['test'] = music_theory_dataset['test'].filter(lambda x: x['subject'] == name).map(convert_dataset)
        return dataset
