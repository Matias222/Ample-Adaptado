import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():

    config = default_config()

    config.result_prefix = ''

    config.tokenizer_paths=["../../../../modelos/Llama-3.2-3B-Instruct"]
    config.model_paths=["../../../../modelos/Llama-3.2-3B-Instruct"]
    config.conversation_templates=['llama-3.2']

    return config
