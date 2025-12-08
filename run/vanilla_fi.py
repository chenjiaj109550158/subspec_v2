import logging
from .app_router import run_app
from .base_builder import GeneratorPipelineBuilder

import torch
from specdecodes.models.generators.naive_fi import NaiveGenerator
from specdecodes.models.utils.cache_utils import create_kv_cache
from specdecodes.models.utils.flashinfer.monkey_patch import apply_flashinfer_kernel_to_llama
from specdecodes.models.utils.flashinfer.cache_manager import FlashInferCache

def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024

class NaiveBuilder(GeneratorPipelineBuilder):
    def __init__(self):
        super().__init__()
        # Base configurations.
        self.vram_limit_gb = None
        self.seed = 0
        self.device = "cuda:0"
        self.dtype = torch.float16
        
        # Model paths.
        self.llm_path = "meta-llama/Llama-3.1-8B-Instruct"
        
        # Generation parameters.
        self.max_length = 128 * 1024
        self.do_sample = False
        self.temperature = 0
        
        # Generator-specific configurations.
        self.generator_kwargs = {
            "prefill_chunk_size": 4096,
            "limit_output_length": None, # limit output length at least 8192 tokens, None: no limit by default
            "page_len": 32,
        }
        
        # Recipe for quantization and offloading.
        self.recipe = None
        
        # Additional configurations.
        self.cache_implementation = "static" 
        self.warmup_iter = 3
        self.compile_mode = None
        
        # Profiling
        self.generator_profiling = True

    def load_kv_cache(self, target_model, draft_model):
        """
        Initialize the KV Cache Pool using FlashInfer logic to match the baseline.
        """
        if self.cache_implementation == "static":
            if self.max_length is not None:
                max_cache_len = self.max_length
            else:
                raise ValueError("max_length should be set for static cache.")
            
            # Initialize FlashInfer KV Pool for Target Model
            past_key_values = FlashInferCache(target_model.config, max_tokens=max_cache_len, PAGE_LEN=self.generator_kwargs["page_len"]).kvCachePool
        else:
            # Fallback to dynamic if needed (though NaiveGenerator now expects Pool)
            past_key_values = create_kv_cache("dynamic")
            
        draft_past_key_values = None
        
        return past_key_values, draft_past_key_values
        
    def load_generator(self, target_model, tokenizer, draft_model=None):
        apply_flashinfer_kernel_to_llama(attention=True, rms_norm=True, swiglu=False, model=target_model)

        generator = NaiveGenerator(
            target_model=target_model,
            tokenizer=tokenizer,
            draft_model=draft_model,
            device=self.device,
            dtype=self.dtype,
            do_sample=self.do_sample,
            temperature=self.temperature,
            profiling_verbose=self.profiling_verbose,
            generator_kwargs=self.generator_kwargs,
        )
        return generator

    def compile_generator(self, generator):
        logging.info(f"Compiling generator with mode: {self.compile_mode}")
        set_torch_compile_config()
        pass
        
if __name__ == "__main__":
    run_app(NaiveBuilder())