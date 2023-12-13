from llama.modeling_llama import LlamaForCausalLM
from llama.tokenization_llama import LlamaTokenizer
from llama.configuration_llama import LlamaConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


MODE = {
    "llama": {"model": LlamaForCausalLM, "tokenizer": LlamaTokenizer, "config": LlamaConfig},
    "skywork": {"model": AutoModelForCausalLM, "tokenizer": AutoTokenizer, "config": AutoConfig}
    }
