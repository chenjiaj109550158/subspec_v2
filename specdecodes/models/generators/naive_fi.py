import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import nvtx

from .base import GeneratorBase
from ..utils.mixin import ProfilingMixin
from ..utils.flashinfer.cache_manager import (
    KvCachePool,
    RequestKvCache,
    KvCacheBatchPosition,
    getKvCacheBatchPosition,
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper


class NaiveGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.prefill_chunk_size = generator_kwargs.get("prefill_chunk_size", None)
        self.limit_output_length = generator_kwargs.get("limit_output_length", None)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        assert self.target_model is not None, "target_model must be provided"

        # Clone input_ids
        input_ids = input_ids.clone()
        batch_size, input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # Prepare kv-cache
        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
        else:
            raise ValueError("past_key_values (KvCachePool) must be provided")

        # Initialize FlashInfer Wrapper
        if not hasattr(self, 'flashinferWrapper'):
            self.flashinferWrapper = FlashinferAttentionWrapper(
                self.target_model.config.num_attention_heads, 
                self.target_model.config.num_key_value_heads, 
                self.target_model.config.hidden_size,
                past_key_values.page_len
            )
        
        self.kvCachePool = past_key_values
        
        request_kv_cache = RequestKvCache(
            kvCachePool=self.kvCachePool,
            page_len=self.kvCachePool.page_len,
            seq_init_len=0
        )

        # Track Sequence Length for Position IDs (Critical for RoPE)
        current_seq_len = 0

        # ── Prefill Stage ─────────────────────────────────────────────────────────────
        with nvtx.annotate("chunked prefill", color="orange"):
            prefill_tokens = input_ids
            prefill_length = prefill_tokens.size(1)
            chunk_size = self.prefill_chunk_size or 4096
            next_token_logits = None
            
            for start in range(0, prefill_length, chunk_size):
                chunk = prefill_tokens[:, start:start + chunk_size]
                chunk_len = chunk.size(1)
                
                # Generate Position IDs for this chunk
                chunk_pos_ids = torch.arange(
                    current_seq_len, 
                    current_seq_len + chunk_len, 
                    dtype=torch.long, 
                    device=input_ids.device
                ).unsqueeze(0)

                request_kv_cache.increment(chunk_len)
                batch_position = getKvCacheBatchPosition(
                    request_kv_caches=[request_kv_cache],
                    mode='tree', 
                    device=input_ids.device,
                    treeTokens=chunk_len,
                )
                self.flashinferWrapper.prepareAttention(
                    'prefill',
                    batch_position,
                    self.kvCachePool.page_len,
                    "NONE", 
                    self.kvCachePool.cache_data[0].dtype,
                )

                if start + chunk_size < prefill_length:
                    self.target_model.prefill_forward(
                        input_ids=chunk,
                        past_key_values=None,
                        use_cache=False,
                        kvCachePool=self.kvCachePool,
                        batch_position=batch_position,
                        mode='prefill', 
                        flashinferWrapper=self.flashinferWrapper,
                        position_ids=chunk_pos_ids, 
                    )
                else:
                    outputs = self.target_model.prefill_forward(
                        input_ids=chunk,
                        past_key_values=None,
                        use_cache=False,
                        logits_to_keep=1,
                        kvCachePool=self.kvCachePool,
                        batch_position=batch_position,
                        mode='prefill', 
                        flashinferWrapper=self.flashinferWrapper,
                        position_ids=chunk_pos_ids,
                    )
                    next_token_logits = outputs.logits
                    del outputs
                
                current_seq_len += chunk_len

        with nvtx.annotate("sample tokens"):
            next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

        with nvtx.annotate("update data"):
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        # ── Decoding Stage ────────────────────────────────────────────────────────────
        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                with nvtx.annotate("llm forward", color="orange"):
                    # Generate Position IDs for the single decoding token
                    pos_ids = torch.tensor(
                        [[current_seq_len]], 
                        dtype=torch.long, 
                        device=input_ids.device
                    )

                    request_kv_cache.increment(1)
                    batch_position = getKvCacheBatchPosition(
                        request_kv_caches=[request_kv_cache],
                        mode='decode', 
                        device=input_ids.device,
                        treeTokens=1,
                    )
                    self.flashinferWrapper.prepareAttention(
                        'decode', 
                        batch_position,
                        self.kvCachePool.page_len,
                        "NONE",
                        self.kvCachePool.cache_data[0].dtype,
                    )
                    outputs = self.target_model(
                        input_ids=next_tokens,
                        past_key_values=None,
                        use_cache=False,
                        kvCachePool=self.kvCachePool,
                        batch_position=batch_position,
                        mode='decode',
                        flashinferWrapper=self.flashinferWrapper,
                        position_ids=pos_ids,
                    )
                    next_token_logits = outputs.logits

                with nvtx.annotate("sample tokens"):
                    next_tokens = self._sample_token(next_token_logits, logits_processor, do_sample)

                with nvtx.annotate("update data"):
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    current_seq_len += 1

                with nvtx.annotate("stopping criteria"):
                    if stopping_criteria(input_ids, None).item():
                        finished = True
        
        request_kv_cache.release()
        return input_ids


class NaiveGenerator(ProfilingMixin, NaiveGeneratorBase):
    pass