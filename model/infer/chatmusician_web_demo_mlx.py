import os
import re
import copy
import time
import logging
import subprocess
from uuid import uuid4
from argparse import ArgumentParser
from pathlib import Path
import gradio as gr
import platform
import sys

# Modify as per the mlx docs: https://github.com/ml-explore/mlx-examples/tree/main/llms
# For conda environment installation: conda install conda-forge::mlx-lm
# Example:
# from mlx_lm import load, generate
# model, tokenizer = load("mistralai/Mistral-7B-Instruct-v0.1")
# response = generate(model, tokenizer, prompt="hello", verbose=True)
from mlx_lm import load, generate

# avoid warning
os.environ['TOKENIZERS_PARALLELISM']='false'

# log_dir
os.makedirs("logs", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
logging.basicConfig(
    filename=f'logs/chatmusician_server_{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))}.log',
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_CKPT_PATH = 'm-a-p/ChatMusician'

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--only_cpu", action="store_true",
                        help="Run demo with CPU only")
    parser.add_argument("--server_port", type=int, default=8888,
                        help="Demo server port.")
    parser.add_argument("--server_name", type=str, default="0.0.0.0",
                        help="Demo server name.")
    parser.add_argument('--title', default=None, type=str)

    args = parser.parse_args()
    return args

def get_uuid():
    return str(uuid4())

def _load_model_tokenizer(args):

    if args.only_cpu:
        device_map = "cpu"
    else:
        device_map = "mps"

    # load_type = (args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype))
    model, tokenizer = load(args.checkpoint_path)
    return model, tokenizer

# todo
def log_conversation(conversation_id, history, messages, response, generate_kwargs):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time()))
    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "history": history,
        "messages": messages,
        "response": response,
        "generate_kwargs": generate_kwargs,
    }
    logging.critical(f"{data}")

def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def convert_history_to_text(task_history):
    history_cp = copy.deepcopy(task_history)
    text = "".join(
        [f"Human: {item[0]} </s> Assistant: {item[1]} </s> " for item in history_cp[:-1] if item[0]]
    )
    text += f"Human: {history_cp[-1][0]} </s> Assistant: "
    return text

def postprocess_abc(text, conversation_id):

    muse_binary = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"

    os.makedirs(f"tmp/{conversation_id}", exist_ok=True)
    abc_pattern = r'(X:\d+\n(?:[^\n]*\n)+)'
    abc_notation = re.findall(abc_pattern, text+'\n')
    print(f'extract abc block: {abc_notation}')
    if abc_notation:
        # Write the ABC text to a temporary file
        tmp_abc = Path(f"tmp/{conversation_id}/{time.time()}.abc")  # xml file
        with open(tmp_abc, "w") as abc_file:
            abc_file.write(abc_notation[0])

        # Convert the temporary ABC file to a MIDI file using abc2midi (requires abc2midi installed)
        tmp_midi = f'tmp/{conversation_id}/{tmp_abc.stem}.mid'
        subprocess.run(["abc2midi", str(tmp_abc), "-o", tmp_midi])

        png_file = f'tmp/{conversation_id}/{tmp_abc.stem}.png'
        wav_file = f'tmp/{conversation_id}/{tmp_abc.stem}.wav'
        svg_file = f'tmp/{conversation_id}/{tmp_abc.stem}.svg'

        print("svg: ", svg_file)
        print("png: ", png_file)
        print("wav: ", wav_file)
        print("midi: ", tmp_midi)
        logging.critical(f"Converted files: tmp/{{conversation_id}}")

        subprocess.run([muse_binary, "--export-to", wav_file, tmp_midi])
        subprocess.run([muse_binary, "--export-to", svg_file, tmp_midi])
        subprocess.run([muse_binary, "--export-to", png_file, tmp_midi])
        # Remove the tmp file
        # tmp_abc.unlink()
        return svg_file, wav_file
    else:
        return None, None

def _launch_demo(args, model, tokenizer):
    logging.critical(f"Inference Model: {args.checkpoint_path}")

    def predict(_chatbot, task_history, temperature, top_p, top_k, repetition_penalty, conversation_id):
        query = task_history[-1][0]
        print("\nUser: " + _parse_text(query))
        # model generation
        messages = convert_history_to_text(task_history)
        start_time = time.time()
        response = generate(model, tokenizer, max_tokens=500,
            temp = float(temperature), repetition_penalty = float(repetition_penalty),
            prompt=_parse_text(query))

        # response = generate(model, tokenizer, prompt=_parse_text(query), verbose=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f">> The output generation took {elapsed_time: .2f} seconds to complete.")

        _chatbot[-1] = (_parse_text(query), _parse_text(response))
        task_history[-1] = (_parse_text(query), response)

        return _chatbot, task_history, response, f"execution time: {elapsed_time: .2f}"

    def process_and_render_abc(_chatbot, task_history, conversation_id):
        svg_file, wav_file = None, None
        try:
            svg_file, wav_file = postprocess_abc(task_history[-1][1], conversation_id)
        except Exception as e:
            logging.error(e)

        if svg_file and wav_file:
            if os.path.exists(svg_file) and os.path.exists(wav_file):
                logging.critical(f"generate: svg: {svg_file} wav: {wav_file}")
                print(f"generate:\n{svg_file}\n{wav_file}")
                _chatbot.append((None, (str(wav_file),)))
                _chatbot.append((None, (str(svg_file),)))
            else:
                logging.error(f"fail to convert: {svg_file[:-4]}.musicxml")
        return _chatbot

    def add_text(history, task_history, text):
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(text, None)]
        return history, task_history, ""

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        conversation_id = gr.State(get_uuid)
        gr.Markdown(
            f"<h1><center>Chat Musician</center></h1>"
        )
        if args.title:
            gr.Markdown(f"<h2><center>{args.title}</center></h2>")
        gr.Markdown("""\
        <center><font size=4><a href="https://ezmonyi.github.io/ChatMusician/">🌐 DemoPage</a>&nbsp |
        &nbsp<a href="https://github.com/hf-lin/ChatMusician">💻 Github</a>&nbsp |
        &nbsp<a href="http://arxiv.org/abs/2402.16153">📖 arXiv</a>&nbsp |
        &nbsp<a href="https://huggingface.co/datasets/m-a-p/MusicTheoryBench">🤗 Benchmark</a>&nbsp |
        &nbsp<a href="https://huggingface.co/datasets/m-a-p/MusicPile">🤗 Pretrain Dataset</a>&nbsp |
        &nbsp<a href="https://huggingface.co/datasets/m-a-p/MusicPile-sft">🤗 SFT Dataset</a>&nbsp |
        &nbsp<a href="https://huggingface.co/m-a-p/ChatMusician">🤖 Chat Model</a>&nbsp |
        &nbsp<a href="https://huggingface.co/m-a-p/ChatMusician-Base">🤖 Base Model</a></center>""")
        gr.Markdown("""\
    <center><font size=4>💡Note: The music clips on this page is auto-converted from abc notations which may not be perfect,
    and we recommend using better software for analysis.</center>""")

        chatbot = gr.Chatbot(label='ChatMusician', elem_classes="control-height", height=750)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        # PG Added for project
        # Add an output TextArea for the generated text that is copyable
        output_text = gr.TextArea(label="Generated Text", interactive=True, lines=5, placeholder="Generated text will appear here")
        # Add an output TextArea for the probabilities
        output_probs = gr.TextArea(label="Probabilities", interactive=True, lines=5, placeholder="Probabilities will appear here")

        with gr.Row():
            submit_btn = gr.Button("🚀 Submit")
            empty_bin = gr.Button("🧹 Clear History")
            # regen_btn = gr.Button("🤔️ Regenerate")
        gr.Examples(
            examples=[
                    ["Craft musical works that follow the given chord alterations: 'Am', 'F', 'C', 'G'"],
                    ["Create music by following the alphabetic representation of the assigned musical structure and the given motif.\n'ABCA';X:1\nL:1/16\nM:2/4\nK:A\n['E2GB d2c2 B2A2', 'D2 C2E2 A2c2']"],
                    ["Create sheet music in ABC notation from the provided text.\nAlternative title: \nThe Legacy\nKey: G\nMeter: 6/8\nNote Length: 1/8\nRhythm: Jig\nOrigin: English\nTranscription: John Chambers"],
                    ["Develop a melody using the given chord pattern.\n'C', 'C', 'G/D', 'D', 'G', 'C', 'G', 'G', 'C', 'C', 'F', 'C/G', 'G7', 'C'"],
                    ["Produce music in compliance with the outlined musical setup in language.\n'Binary', 'Sectional: Verse/Chorus'"],
                ],
            inputs=query
            )
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.2,
                                minimum=0.0,
                                maximum=10.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=0.9,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=40,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history], queue=False).then(
            predict,
            inputs = [chatbot, task_history, temperature, top_p, top_k, repetition_penalty, conversation_id],
            outputs = [chatbot, task_history, output_text, output_probs],
            show_progress=True,
            queue=True
        ).then(process_and_render_abc, [chatbot, task_history, conversation_id], [chatbot])
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)

        gr.Markdown(
                "Disclaimer: The model can produce factually incorrect output, and should not be relied on to produce "
                "factually accurate information. The model was trained on various public datasets; while great efforts "
                "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
                "biased, or otherwise offensive outputs.",
                elem_classes=["disclaimer"],
            )

    demo.queue().launch(
        server_port=args.server_port,
        server_name=args.server_name,
        share=True,
    )

def main():
    args = _get_args()

    model, tokenizer = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer)

if __name__ == '__main__':
    main()
