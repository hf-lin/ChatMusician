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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

os.environ['QT_QPA_PLATFORM']='offscreen'
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# log_dir
os.makedirs("logs", exist_ok=True)
os.makedirs("tmp", exist_ok=True)
logging.basicConfig(
    filename=f'logs/chatmusician_server_{time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))}.log',
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_CKPT_PATH = '/data/hanfeng/chat_musician_models/epoch-2-step-285220/'

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

    parser.add_argument('--load_in_8bit', action='store_true', 
                        help="Load the LLM in the 8bit mode")
    parser.add_argument('--torch_dtype', default="float16", type=str, choices=["auto", "bfloat16", "float16", "float32"], 
                        help="Load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.")

    args = parser.parse_args()
    return args

def get_uuid():
    return str(uuid4())

def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.only_cpu:
        device_map = "cpu"
    else:
        device_map = "cuda"
    load_type = (
            args.torch_dtype if args.torch_dtype in ["auto", None]
            else getattr(torch, args.torch_dtype)
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        load_in_8bit=args.load_in_8bit,
        device_map=device_map,
        torch_dtype=load_type, 
        trust_remote_code=True,
        resume_download=True,
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

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

# todo
def postprocess_abc(text, conversation_id):
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

        # Convert xml to SVG and WAV using MuseScore (requires MuseScore installed)
        svg_file = f'tmp/{conversation_id}/{tmp_abc.stem}.svg'
        wav_file = f'tmp/{conversation_id}/{tmp_abc.stem}.mp3'
        subprocess.run(["./MuseScore-4.1.1.232071203-x86_64.AppImage", "-f", "-o", svg_file, tmp_midi])
        subprocess.run(["./MuseScore-4.1.1.232071203-x86_64.AppImage", "-f", "-o", wav_file, tmp_midi])

        # Remove the tmp file
        # tmp_abc.unlink()
        return svg_file, wav_file
    else:
        return None, None


def _launch_demo(args, model, tokenizer):
    logging.critical(f"Inference Model: {args.checkpoint_path}")

    def predict(_chatbot, task_history, temperature, top_p, top_k, repetition_penalty, conversation_id):
        query = task_history[-1][0]
        print("User: " + _parse_text(query))
        # model generation
        messages = convert_history_to_text(task_history)
        inputs = tokenizer(messages, return_tensors="pt", add_special_tokens=False)
        generation_config = GenerationConfig(
            temperature=float(temperature), 
            top_p = float(top_p), 
            top_k = top_k, 
            repetition_penalty = float(repetition_penalty),
            max_new_tokens=1536,
            min_new_tokens=5,
            do_sample=True,
            num_beams=1,
            num_return_sequences=1
        )
        response = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs['attention_mask'].to(model.device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
                )
        response = tokenizer.decode(response[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        _chatbot[-1] = (_parse_text(query), _parse_text(response))
        task_history[-1] = (_parse_text(query), response)
        # log
        log_conversation(conversation_id, task_history, messages, _chatbot[-1][1], generation_config.to_json_string())
        return _chatbot, task_history
        
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
            gr.Markdown(
                f"<h2><center>{args.title}</center></h2>"
            )
        gr.Markdown("""\
        <center><font size=4>Chat-Musician <a href="https://huggingface.co/m-a-p/ChatMusician-v1-sft-78k">🤗</a>&nbsp ｜ 
        &nbsp<a href="https://github.com/a43992899/Chat-Musician">Github</a></center>""")
        gr.Markdown("""\
    <center><font size=4>💡Note: The music clips on this page is auto-converted by abc notations which may not be perfect, 
    and we recommend using better software for analysis.</center>""")

        chatbot = gr.Chatbot(label='Chat-Musician', elem_classes="control-height", height=750)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])
        
        with gr.Row():
            submit_btn = gr.Button("🚀 Submit (发送)")
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            # regen_btn = gr.Button("🤔️ Regenerate (重试)")
        gr.Examples(
            examples=[
                    ["Utilize the following musical structure as a guide to shape your composition.\n'Binary', 'Sectional: Verse/Chorus'"],
                    ["Create music by following the alphabetic representation of the assigned musical structure and the given motif.\n'ABCA';X:1\nL:1\/16\nM:2\/4\nK:A\n['E2GB d2c2 B2A2', 'D2 C2E2 A2c2']"],
                    ["Create sheet music in ABC notation from the provided text.\nAlternative title: \nThe Legacy\nKey: G\nMeter: 6/8\nNote Length: 1/8\nRhythm: Jig\nOrigin: English\nTranscription: John Chambers"],
                    ["Develop a melody using the given chord pattern.\n'C', 'C', 'G/D', 'D', 'G', 'C', 'G', 'G', 'C', 'C', 'F', 'C/G', 'G7', 'C'"]
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
            [chatbot, task_history, temperature, top_p, top_k, repetition_penalty, conversation_id], 
            [chatbot, task_history], 
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