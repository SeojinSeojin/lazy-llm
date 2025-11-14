import os
import torch
import gc
import getpass
import shutil
import tempfile
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from dotenv import load_dotenv

class LLM:
    """
      Class LLM: it is the parent class of all the LLM Models we are going to use for this project
      params: it just takes the parameters like model names and basic configuration of the LLMS
    """
    def __init__(self, model_name : str, temperature : float, max_tokens : int, top_p : float, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
    
    def __repr__(self):
        return f'LLM(model = {self.model_name}, temperature = {self.temperature}, max token = {self.max_tokens}, top p = {self.top_p})'
    
class Local_LLM(LLM):
    """

    """
    def __init__(self, model_name : str, temperature : float, max_tokens : int, top_p : float, cache : bool,  **kwargs):
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.cache = cache
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.quantization = kwargs.get('quantization', False)
        self.quant_bits = kwargs.get('nbits', 8)
        
        if self.quantization and self.quant_bits == 4:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_use_double_quant = True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype = torch.bfloat16
            )
        
        elif self.quantization and self.quant_bits == 8:
            self.bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="int8",
                bnb_8bit_compute_dtype=torch.bfloat16  
            )

        self.pipe = self.get_pipeline()

    def get_pipeline(self):
        if not self.cache:
            self.temp_dir = tempfile.mkdtemp()
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir = self.temp_dir)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map= "auto")

        pipe = pipeline(
         "text-generation",
         model=self.model,
         tokenizer = self.tokenizer,
         device_map = "auto",
        )

        return pipe

    def get_params(self, model=None) -> str:
        total_params = sum(p.numel() for p in self.model.parameters())
        if total_params >= 1e9:
            return f"{total_params / 1e9:.1f}B"
        elif total_params >= 1e6:
            return f"{total_params / 1e6:.1f}M"
        else:
            return str(total_params)

    def get_model_with_quantization(self) -> HuggingFacePipeline:
        if not self.quantization:
            raise Exception("Your model does not have necessary quantization config setup")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config = self.bnb_config)
        pipe = pipeline(
               "text-generation",
                model= self.model,
                tokenizer= self.tokenizer,
                max_new_tokens= self.max_tokens, 
                temperature= self.temperature, 
                top_p= self.top_p, 
                do_sample = True  
               )

    def unload_model(self): 
        del self.model
        gc.collect()
        if hasattr(self, "temp_dir") and getattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass

    def generate_text(self, messages: list) -> str:
        """Generate text for chat-style messages (local HuggingFace)."""
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=self.max_tokens, do_sample=True,
                            temperature=self.temperature, top_p=self.top_p)
        return outputs[0]['generated_text'][len(prompt):]  # remove prompt prefix

# --------------------------------------------------------------------
# API LLM (OpenAI / Gemini)
# --------------------------------------------------------------------
class API_LLM(LLM):
    """API-based LLM (Google Gemini, OpenAI GPT)."""

    def __init__(self, model_name, temperature, max_tokens, top_p, **kwargs):
        super().__init__(model_name, temperature, max_tokens, top_p)
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize API client safely."""
        load_dotenv()
        if "gemini" in self.model_name.lower():
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY: ")
            self.client = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )
        elif "gpt" in self.model_name.lower():
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI_API_KEY: ")
            self.client = OpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
        else:
            raise ValueError(f"Unsupported API model: {self.model_name}")

    def generate_text(self, messages):
        """Generate text from structured messages using the API client."""
        if self.client is None:
            self._init_client()  # ensure client exists

        # Convert message dicts to chat format
        text_prompt = "\n".join([f"[{m['role'].upper()}] {m['content']}" for m in messages])

        try:
            # --- Gemini ---
            if isinstance(self.client, ChatGoogleGenerativeAI):
                response = self.client.invoke(text_prompt)
                # Some versions of LangChain return 'str', others return an object
                if isinstance(response, str):
                    return response.strip()
                elif hasattr(response, "content") and len(response.content) > 0:
                    return getattr(response.content[0], "text", str(response.content[0]))
                else:
                    return str(response)

            # --- OpenAI ---
            elif isinstance(self.client, OpenAI):
                response = self.client.invoke(text_prompt)
                if isinstance(response, str):
                    return response.strip()
                elif hasattr(response, "content"):
                    return str(response.content)
                elif hasattr(response, "output_text"):
                    return str(response.output_text)
                return str(response)

            else:
                raise RuntimeError("Unknown API client type.")

        except Exception as e:
            raise RuntimeError(f"API generate_text failed: {e}")

    def get_params(self, *_, **__):
        provider = (
            "Google" if "gemini" in self.model_name.lower()
            else "OpenAI" if "gpt" in self.model_name.lower()
            else "Unknown"
        )
        return f"{provider} API model ({self.model_name})"

    def unload_model(self):
        self.client = None
        gc.collect()
        print(f"[UNLOAD] API model {self.model_name} released.")