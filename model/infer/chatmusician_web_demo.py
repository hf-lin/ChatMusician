import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from transformers import GenerationConfig
from peft import PeftModel
import datetime
from threading import Event, Thread
from uuid import uuid4
import queue
import argparse
import logging
import gradio as gr

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=f'logs/chatmusician_server_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.log',
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
message_queue = queue.Queue()

max_len = 2048
min_new_tokens = 10
max_new_tokens = 1536
max_src_len = 1530

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_token_ids = [tokenizer.eos_token_id]
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def convert_history_to_text(history):
    text = "".join(
        [f"Human: {item[0]} </s> Assistant: {item[1]} </s> " for item in history[:-1]]
    )
    text += f"Human: {history[-1][0]} </s> Assistant: "
    return text

def log_conversation_local(conversation_id, history, messages, response, generate_kwargs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "history": history,
        "messages": messages,
        "response": response,
        "generate_kwargs": generate_kwargs,
    }
    logging.critical(f"{data}")

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = convert_history_to_text(history)
    print("input text: \n"+messages)
    
    inputs = tokenizer(messages, return_tensors="pt", add_special_tokens=False)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True,
        num_beams=1,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens
    )
    stream_complete = Event()

    def generate_and_signal_complete():
        response = model.generate(
            input_ids=inputs["input_ids"].to(model.device),
            attention_mask=inputs['attention_mask'].to(model.device),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config,
            streamer=streamer
            )
        response = tokenizer.decode(response[0])
        message_queue.put(response)
        stream_complete.set()

    def log_after_stream_complete():
        stream_complete.wait()
        while True:
            response = message_queue.get()
            if response:
                log_conversation_local(
                    conversation_id,
                    history,
                    messages,
                    response,
                    {
                        "top_k": top_k,
                        "top_p": top_p,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        "do_sample": True
                    },
                )
                break  # Exit the thread when a message is received

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    t2 = Thread(target=log_after_stream_complete)
    t2.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    # print(streamer)
    for new_text in streamer:
        # if new_text == '<br>':
        #     continue
        partial_text += new_text
        # partial_text.r
        print(partial_text)
        history[-1][1] = partial_text
        yield history


def get_uuid():
    return str(uuid4())


def init(args):
    if args.lora_model:
        logging.critical(f"Inference Peft Model: {args.lora_model}")
    else:
        logging.critical(f"Inference Model: {args.base_model}")
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        conversation_id = gr.State(get_uuid)
        gr.Markdown(
            f"<h1><center>Chat Musician ({args.title})</center></h1>"
        )
        chatbot = gr.Chatbot().style(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                ).style(container=False)
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
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
        with gr.Row():
            gr.Markdown(
                "Disclaimer: The model can produce factually incorrect output, and should not be relied on to produce "
                "factually accurate information. The model was trained on various public datasets; while great efforts "
                "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
                "biased, or otherwise offensive outputs.",
                elem_classes=["disclaimer"],
            )
        with gr.Row():
            gr.Markdown(
                "[Privacy policy](https://gist.github.com/samhavens/c29c68cdcd420a9aa0202d0839876dac)",
                elem_classes=["disclaimer"],
            )

        submit_event = msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        submit_click_event = submit.click(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue(max_size=128, concurrency_count=2)

    gr.close_all()

    demo.launch(server_name=args.server_name, server_port=args.server_port, inline=False, share=True)

if __name__ == "__main__":
    global model
    global tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='m-a-p/ChatMusician-v1-sft-78k', type=str, required=True)
    parser.add_argument('--lora_model', default=None, type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Demo server name.")
    parser.add_argument("--server_port", type=int, default=8888, help="Demo server port.")
    parser.add_argument('--title', default=None, type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--load_in_8bit', action='store_true', help="Load the LLM in the 8bit mode")
    parser.add_argument('--torch_dtype', default="float16", type=str, choices=["auto", "bfloat16", "float16", "float32"], 
                        help="Load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.")
    args = parser.parse_args()

    load_type = (
            args.torch_dtype if args.torch_dtype in ["auto", None]
            else getattr(torch, args.torch_dtype)
        )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=load_type,
        trust_remote_code=True,
        device_map="auto"
    )
    if args.lora_model is not None:
        print("loading peft model")
        model = PeftModel.from_pretrained(
            base_model, 
            args.lora_model, 
            load_in_8bit=args.load_in_8bit,
            torch_dtype=load_type, 
            device_map='auto'
        )
        print(f"Successfully loaded peft model {args.lora_model} into memory")
    else:
        model = base_model
        print(f"Successfully loaded the model {args.base_model} into memory")

    if args.tokenizer_path is None:
        args.tokenizer_path = args.lora_model
        if args.lora_model is None:
            args.tokenizer_path = args.base_model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenzier_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
    if model_vocab_size != tokenzier_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenzier_vocab_size)
    
    init(args)