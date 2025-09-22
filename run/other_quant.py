import torch
from run.app_router import run_app
from run.vanilla import NaiveBuilder

from specdecodes.helpers.recipes.quant.hqq_4bit import Recipe
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, AwqConfig

class NaiveBuilder(NaiveBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = None
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Model paths.
        self.llm_path = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4" #"meta-llama/Llama-3.1-8B-Instruct"
        
        # Generation parameters.
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {
            "prefill_chunk_size": 256,
        }
        
        # Recipe for quantization and offloading.
        self.recipe = Recipe()
        
        # Additional configurations.
        self.cache_implementation = "static"
        self.warmup_iter = 3
        self.compile_mode = "max-autotune"
        
        # Profiling
        self.generator_profiling = True
        
    def load_model_and_tokenizer(self, model_path: str):
        """
        Load a model and tokenizer from the specified model path.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use CPU if an offloader is provided via recipe; otherwise use the desired device.
        device_map = 'cpu' if (self.recipe and self.recipe.offloader) else self.device
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            device_map="auto",
            # _attn_implementation="sdpa",
            quantization_config=GPTQConfig(bits=4, backend="triton")
            # quantization_config=AwqConfig(
            #     bits=4,
            #     fuse_max_seq_len=2048,
            #     do_fuse=True,
            # )
        )
    
        return model, tokenizer
        
if __name__ == "__main__":
    run_app(NaiveBuilder())