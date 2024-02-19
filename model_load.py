import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os


class Model:
    def __init__(self) -> None:
        self.base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self.eval_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            add_bos_token=True,
            trust_remote_code=True
        )
        self.ft_model = PeftModel.from_pretrained(
            self.base_model,
            model_id = os.path.join('Mistral-instruct-v1.0-4bit-CTadaptor')
        )
