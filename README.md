# Chat-Musician

## Introduction

If artificial intelligence cannot effectively model the delicate, emotional, and universal human language —— music, then it might not be deemed AGI (Artificial General Intelligence). Here, we introduce 🎼 ChatMusician, an open-source large language model (LLM) family that integrates intrinsic musical abilities. 🎼 ChatMusician, built upon the LLaMA2 architecture, has been continuously pre-trained and fine-tuned using a meticulously curated music-language corpus named MusicPile. We demonstrate that 🎼 ChatMusician can perform various conditional symbolic music generation tasks. It learns to follow diverse musical instructions and compose coherent, well-structured music pieces. Subjective evaluations indicate that ChatMusician outperforms GPT4 across a variety of conditional music generation tasks. Moreover, on our newly collected college-level music understanding benchmark, MusicTheoryBench, our model surpasses LLaMA2 base and ChatGPT3.5 by a large margin. While the LLMs that we investigated demonstrate impressive capabilities in musical knowledge, we find that musical reasoning is still an unsolved task. 🎼 ChatMusician, or even the SOTA GPT4 only surpasses random baseline by a small margin. Our work reveals that LLMs can be an excellent repository for music, an important aspect of humanity’s cultural and creative language, but there remains significant territory to be conquered. Code, data, model, and benchmark are open-sourced.

## Models

|                             | HuggingFace model                                                               |  
|-----------------------------|---------------------------------------------------------------------------------|
| continuously pretrain model | [ChatMusician-v1-base](https://huggingface.co/m-a-p/ChatMusician-v1-base)       |
| sft model                   | [ChatMusician-v1-sft-78k](https://huggingface.co/m-a-p/ChatMusician-v1-sft-78k) |
|                             | [ChatMusician-v1-sft-1M](https://huggingface.co/m-a-p/ChatMusician-v1-sft-1M)   |

## Dataset

The models are trained on the [🤗 ChatMusician-v1-pt Dataset](https://huggingface.co/datasets/m-a-p/ChatMusician-v1-pt), which is the first pretraining corpus for developing musical abilities in large language models. Check out the dataset card for more details.

## Inference

### web demo (with audio)

You need to install abc2midi and musescore.
```
sudo apt-get update
sudo apt-get install abcmidi
sudo apt-get install musescore
```
Then launch a gradio demo:
```bash
cd ChatMusician/
python model/infer/chatmusician_web_demo_audio.py -c "ChatMusician-v1-sft-78k" --server_port 8888
```
Prompt example:
```
Using ABC notation, recreate the given text as a musical score.
Meter C
Notes The parts are commonly interchanged.
Transcription 1997 by John Chambers
Key D
Note Length 1/8
Rhythm reel
```
![chatmusician web demo](model/res/prompt1.png)

### inferece locally

```bash
cd Chat-Musician/
python model/infer/predict.py --base_model {merged_model_path} --with_prompt --interactive
```
Note: with `--with_prompt`, input text will be converted to chat format.

## Training

### Requirements

- Python 3.8 and above
- Pytorch 2.0 and above are recommended
- CUDA 11.4 and above are recommended
- Deepspeed 0.10 and above are recommended

Python dependency installation:
```
pip install -r requirements.txt 
```

### Data Preprocessing

```bash
## preprocess continue pretraining data
python model/train/data_preprocess.py \
    -t $TOKENIZER_PATH \
    -i $DATA_FILE \
    -o $OUTPUT_DIR 
```

### Pretraining or Supervised Fine-tuning

run `model/train/train.sh`

## Merge Peft Model

```bash
cd Chat-Musician/
python model/train/merge.py --ori_model_dir {base_model} --model_dir {lora_ckpt_path} --output_dir {output_path}
```
