# Evaluation with Massive Multitask Language Understanding (MMLU) Benchmark and MusicTheoryBench

## Installation

Below are the steps for quick installation and datasets preparation.

```Python
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
pip install -e .
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

## ️Evaluation

After ensuring that OpenCompass is installed correctly according to the above steps and the datasets are prepared, you can evaluate the performance of our ChatMusician model on the MMLU and MusicTheoryBench datasets using the following command:

```bash
python run.py configs/eval_chat_musician_7b.py
```