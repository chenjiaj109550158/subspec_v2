import torch
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria
import logging
import nvtx

from .base import GeneratorBase
from ..utils.mixin import SDProfilingMixin
from ..utils.flashinfer.cache_manager import (
    KvCachePool,
    RequestKvCache,
    KvCacheBatchPosition,
    getKvCacheBatchPosition,
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper

class ClassicSeqFiGeneratorBase(GeneratorBase):
    def __init__(self, generator_kwargs, *model_args, **kwargs):
        super().__init__(*model_args, **kwargs)
        self.generator_kwargs = generator_kwargs or {}
        self.prefill_chunk_size = self.generator_kwargs.get("prefill_chunk_size", None)
        self.limit_output_length = generator_kwargs.get("limit_output_length", None)

    def init_cuda_graph_runner(self, device, kvCachePool=None):
        """
        Initialize the CUDA graph runner for the draft model if available.
        """
        if hasattr(self.draft_model, 'init_cuda_graph_runner') and callable(self.draft_model.init_cuda_graph_runner):
            self.draft_model.init_cuda_graph_runner(device=device)
    
    def _truncate_cache(self, request_kv_cache, target_len):
        """
        Efficiently truncates the cache to target_len.
        """
        # Primary method: crop (now fixed in cache_manager to handle pages correctly)
        if hasattr(request_kv_cache, 'crop'):
            request_kv_cache.crop(target_len)
        else:
            # Fallback
            current_len = request_kv_cache.get_seq_length()
            if isinstance(current_len, torch.Tensor):
                current_len = current_len.item()
            if current_len > target_len:
                request_kv_cache.decrement(current_len - target_len)

    def _speculate(self, input_ids, request_kv_cache):
        return self.draft_model.speculate(
            input_ids,
            request_kv_cache=request_kv_cache,
            flashinferWrapper=self.flashinferWrapper,
        )

    def _tree_decoding(self, draft_ids, request_kv_cache, cache_position, device):
        seq_len = draft_ids.shape[1]
        
        with nvtx.annotate("verify forward", color="red"):
            # Target start offset is the index of the last token in input_ids
            # We want to OVERWRITE the Anchor.
            target_start_offset = cache_position[0].item() - 1
            if target_start_offset < 0: target_start_offset = 0
            
            # 1. Truncate/Align Cache to target_start_offset
            self._truncate_cache(request_kv_cache, target_start_offset)
            
            rollback_start_offset = target_start_offset

            # 2. Append the draft sequence (Anchor + New Tokens)
            request_kv_cache.increment(seq_len)

            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode='tree', 
                device=device,
                treeTokens=seq_len,
            )
            
            self.flashinferWrapper.prepareAttention(
                'prefill', 
                batch_position,
                request_kv_cache.kvCachePool.page_len,
                "NONE", 
                request_kv_cache.kvCachePool.cache_data[0].dtype,
            )

            start_pos = rollback_start_offset
            chunk_pos_ids = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=device).unsqueeze(0)

            outputs = self.target_model.prefill_forward(
                input_ids=draft_ids,
                past_key_values=None,
                use_cache=False,
                kvCachePool=request_kv_cache.kvCachePool,
                batch_position=batch_position,
                mode='prefill', 
                flashinferWrapper=self.flashinferWrapper,
                logits_to_keep=seq_len, 
                position_ids=chunk_pos_ids
            )
            
        return outputs.logits, rollback_start_offset

    def _verify(self, draft_ids, target_logits):
        target_tokens = torch.argmax(target_logits, dim=-1) # [1, Seq_Len]
        
        draft_seq = draft_ids[0, 1:]
        target_seq = target_tokens[0, :-1]
        
        min_len = min(draft_seq.shape[0], target_seq.shape[0])
        if min_len == 0:
            bonus_token = target_tokens[0, 0:1].unsqueeze(0)
            sampled_tokens = bonus_token
            valid_indices = torch.tensor([0], device=draft_ids.device, dtype=torch.long)
            return sampled_tokens, valid_indices, (0, 0)

        matches = (draft_seq[:min_len] == target_seq[:min_len])
        
        accept_len = 0
        for m in matches:
            if m:
                accept_len += 1
            else:
                break
        
        bonus_token = target_tokens[0, accept_len].unsqueeze(0).unsqueeze(0)
        
        accepted_tokens = draft_ids[0, 1 : 1 + accept_len].unsqueeze(0)
        sampled_tokens = torch.cat([accepted_tokens, bonus_token], dim=-1)
        
        valid_indices = torch.arange(0, 1 + accept_len, device=draft_ids.device, dtype=torch.long)
        seq_len = draft_ids.shape[1] - 1
        
        return sampled_tokens, valid_indices, (seq_len, accept_len)

    def _generate(
        self,
        input_ids: torch.LongTensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessorList,
        do_sample: bool,
        **model_kwargs,
    ):
        """
        Generate sequence of tokens with speculative decoding (Sequence Style).

        Args:
            input_ids (torch.LongTensor): The input token IDs. 
            stopping_criteria (StoppingCriteria): The criteria to stop the generation.
            logits_processor (LogitsProcessor): The processor to modify the logits.
            do_sample (bool): Whether to sample tokens during generation.

        Returns:
            input_ids (torch.LongTensor): The generated token IDs.
        """
        assert self.target_model is not None, "target_model must be provided"
        assert self.draft_model is not None, "draft_model must be provided"
        assert self.tokenizer is not None, "tokenizer must be provided"
        
        # * clone input_ids
        input_ids = input_ids.clone()
        batch_size, org_input_len = input_ids.shape
        assert batch_size == 1, "Only support batch_size=1 for now."

        # * prepare kv-cache
        # Raise error if max_length not set while using static cache
        if stopping_criteria.max_length is None:
            if self.cache_implementation == "static":
                raise ValueError(
                    "max_length is not set. Only 'dynamic' kv-cache is supported when max_length is unspecified."
                )
            
        if model_kwargs.get("past_key_values") is not None:
            past_key_values = model_kwargs["past_key_values"]
            if model_kwargs.get("draft_past_key_values") is not None:
                draft_past_key_values = model_kwargs["draft_past_key_values"]
            else:
                raise ValueError("draft_past_key_values must be provided")
        else:
            raise ValueError("past_key_values must be provided")

        # * prefill stage
        with nvtx.annotate("chunked prefill", color="orange"):
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
            draft_request_kv_cache = RequestKvCache(
                kvCachePool=draft_past_key_values,
                page_len=draft_past_key_values.page_len,
                seq_init_len=0
            )

            current_kv_len = 0
            prefill_tokens = input_ids[:, current_kv_len:]
            prefill_length = prefill_tokens.size(1)
            chunk_size = prefill_length if self.prefill_chunk_size is None else min(prefill_length, self.prefill_chunk_size)
            
            for start in range(0, prefill_length, chunk_size):
                chunk = prefill_tokens[:, start:start + chunk_size]
                num_new_tokens = chunk.size(1)
                
                request_kv_cache.increment(num_new_tokens)
                
                batch_position = getKvCacheBatchPosition(
                    request_kv_caches=[request_kv_cache],
                    mode='tree', 
                    device=input_ids.device,
                    treeTokens=num_new_tokens,
                )
                self.flashinferWrapper.prepareAttention(
                    'prefill',
                    batch_position,
                    self.kvCachePool.page_len,
                    "NONE", 
                    self.kvCachePool.cache_data[0].dtype,
                )
                
                chunk_pos_ids = torch.arange(
                    current_kv_len, 
                    current_kv_len + num_new_tokens, 
                    dtype=torch.long, 
                    device=input_ids.device
                ).unsqueeze(0)

                self.target_model.prefill_forward(
                    input_ids=chunk,
                    position_ids=chunk_pos_ids, 
                    past_key_values=None,
                    use_cache=False,
                    kvCachePool=self.kvCachePool,
                    batch_position=batch_position,
                    mode='prefill', 
                    flashinferWrapper=self.flashinferWrapper,
                )
                
                current_kv_len += num_new_tokens

        cache_position = torch.tensor([input_ids.shape[1]], device=input_ids.device)

        with nvtx.annotate("decoding"):
            finished = False
            while not finished:
                # * speculate
                with nvtx.annotate("speculate", color="cyan"):
                    draft_ids = self._speculate(input_ids, draft_request_kv_cache)

                # * verify forward
                with nvtx.annotate("verify forward", color="orange"):
                    target_logits, rollback_start_offset = self._tree_decoding(draft_ids, request_kv_cache, cache_position, input_ids.device)

                # * verify logic
                with nvtx.annotate("verify logic"):
                    sampled_tokens, valid_indices, (seq_len, accept_len) = self._verify(draft_ids, target_logits)
                    new_tokens = sampled_tokens
                    
                # * crop kv and update data
                with nvtx.annotate("crop kv and update data"):
                    # Check for EOS and truncate if present
                    if self.tokenizer.eos_token_id is not None:
                        eos_mask = (new_tokens == self.tokenizer.eos_token_id)
                        if eos_mask.any():
                            eos_indices = torch.nonzero(eos_mask[0], as_tuple=True)[0]
                            if len(eos_indices) > 0:
                                cut_idx = eos_indices[0].item()
                                new_tokens = new_tokens[:, :cut_idx+1]

                    input_ids = torch.cat([input_ids, new_tokens], dim=-1)
                    cache_position += new_tokens.shape[1]
                    
                    # Rollback Target Cache
                    final_target_len = rollback_start_offset + new_tokens.shape[1]
                    self._truncate_cache(request_kv_cache, final_target_len)
                    
                    # Rollback Draft Cache
                    desired_draft_len = input_ids.shape[1] - 1
                    self._truncate_cache(draft_request_kv_cache, desired_draft_len)

                # * check stopping criteria
                with nvtx.annotate("stopping criteria"):
                    if stopping_criteria(input_ids, None).item():
                        finished = True
                        
        request_kv_cache.release()   
        draft_request_kv_cache.release()  
        return input_ids
    
class ClassicSeqFiGenerator(SDProfilingMixin, ClassicSeqFiGeneratorBase):
    pass