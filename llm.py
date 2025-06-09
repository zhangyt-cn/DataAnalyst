

import os
from typing import List, Optional
import openai
from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

class OpenAILLM:
    """
    OpenAI API based LLM, including OpenAI and other interface-compatible models
    """
    def __init__(self, config: dict):
        
        self.client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"]
        )
        self.config = config
       

    def generate_response(self, messages) -> Optional[str]:
        max_tries = 5
        while True:
            try:
                response = self.client.chat.completions.create(
                messages=messages,
                **self.config
                )
                return response.choices[0].message.content
            except openai.OpenAIError as e:
                print(f"OpenAI API Error: {e}")
                max_tries -= 1
                if max_tries < 0:
                    return None
            except Exception as e:
                print(f"Unexpected Error: {str(e)}")
                max_tries -= 1
                if max_tries < 0:
                    return None
               
        
class CustomLLM:
    """
    Local LLM especially fine tuned.
    """
    def __init__(self, model_path: str, lora_path: str=None, config: dict=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        self.model = PeftModel.from_pretrained(self.model, model_id=lora_path)
        pass

    def generate_response(self, messages: List[dict]) -> str:

        format_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([format_text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=1024)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response