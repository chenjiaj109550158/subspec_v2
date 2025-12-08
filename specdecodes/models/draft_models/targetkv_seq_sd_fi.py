import torch
import nvtx
import json

from .base import DraftModelBase

from ..utils.flashinfer.cache_manager import (
    KvCachePool,
    RequestKvCache,
    KvCacheBatchPosition,
    getKvCacheBatchPosition,
    FlashInferCache
)
from ..utils.flashinfer.attention_wrapper import FlashinferAttentionWrapper
from ..utils.compresskv.monkey_patch import CaptureAttentionContext

class TargetkvSeqFiDraftModel(DraftModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize placeholders for captured data
        self.latest_captured_queries = None
        self.latest_captured_rope_queries = None
        self.important_layers = None
        self.important_heads = None
    
    def forward(self, input_ids, with_softmax=False, *model_args, **kwargs):
        logits = self.model(input_ids, *model_args, **kwargs).logits
        if with_softmax:
            logits = torch.softmax(logits/self.draft_params.temperature, dim=-1)
        return logits
    
    def init_cuda_graph_runner(
        self, 
        device: torch.device
    ):
        """
        Allocate *fixed‑size* staging buffers for a single‑batch, single-token
        decoding step and capture it inside a CUDA Graph.

        Call once (e.g. the very first time `speculate` is invoked).
        """
        print("Draft model initializing CUDA Graph runner for Sequence Decoding...", flush=True)
        if hasattr(self, "graph") and self.graph is not None:
            return

        self.decode_chunk_size = 1
        self.graph              = None
        self.output_buffer      = None
        self.model.eval()   
        kvCachePool = self.kvCachePool

        # ── staging buffers ───────────────────────────────────────
        B = 1
        L = self.decode_chunk_size
        
        self.input_ids_buf    = torch.zeros((B, L),  dtype=torch.long,  device=device)
        self.position_ids_buf = torch.zeros((B, L),  dtype=torch.long,  device=device)

        self.seq_indptr_buf        = torch.zeros((B + 1,),          dtype=torch.int32, device=device)
        self.kv_page_indptr_buf    = torch.zeros((B + 1,),          dtype=torch.int32, device=device)
        self.kv_page_indices_buf   = torch.zeros((kvCachePool.max_pages,), dtype=torch.int32, device=device)
        self.kv_last_page_len_buf  = torch.zeros((B,),              dtype=torch.int32, device=device)
        self.batch_indices_buf     = torch.zeros((L,),              dtype=torch.int32, device=device)
        self.positions_buf         = torch.zeros((L,),              dtype=torch.int32, device=device)

        self.batch_position = KvCacheBatchPosition(
            seq_indptr       = self.seq_indptr_buf,
            kv_page_indptr   = self.kv_page_indptr_buf,
            kv_page_indices  = self.kv_page_indices_buf,
            kv_last_page_len = self.kv_last_page_len_buf,
            batch_indices    = self.batch_indices_buf,
            positions        = self.positions_buf,
        )

        # ── Flash‑Infer wrapper (shared weights, no extra allocation) ──
        if not hasattr(self, "flashinferWrapper"):
            self.flashinferWrapper = FlashinferAttentionWrapper(
                self.model.config.num_attention_heads, 
                self.model.config.num_key_value_heads, 
                self.model.config.hidden_size, 
                kvCachePool.page_len
            )

        # ── Workspace Warmup & Buffer Prefill ─────────────────────
        # Force FlashInfer to allocate a workspace large enough for the MAXIMUM possible sequence.
        max_possible_tokens = kvCachePool.max_pages * kvCachePool.page_len
        warmup_request = RequestKvCache(kvCachePool, kvCachePool.page_len, seq_init_len=0)
        warmup_request.increment(max_possible_tokens) 

        warmup_bp = getKvCacheBatchPosition(
            request_kv_caches=[warmup_request],
            mode='decode', 
            device=device,
            treeTokens=1,
        )

        self.flashinferWrapper.prepareAttention(
            'decode',
            warmup_bp, 
            kvCachePool.page_len,
            "NONE",
            kvCachePool.cache_data[0].dtype
        )
        warmup_request.release()

        # Pre-fill buffers with valid dummy data (1 token)
        dummy_request = RequestKvCache(kvCachePool, kvCachePool.page_len, seq_init_len=0)
        dummy_request.increment(1) 

        dummy_bp = getKvCacheBatchPosition(
            request_kv_caches=[dummy_request],
            mode='decode', 
            device=device,
            treeTokens=1,
        )

        self.seq_indptr_buf.copy_(dummy_bp.seq_indptr)
        self.kv_page_indptr_buf.copy_(dummy_bp.kv_page_indptr)
        self.kv_last_page_len_buf.copy_(dummy_bp.kv_last_page_len)
        self.batch_indices_buf.copy_(dummy_bp.batch_indices)
        self.positions_buf.copy_(dummy_bp.positions)
        n_pages = dummy_bp.kv_page_indptr[-1].item()
        self.kv_page_indices_buf[:n_pages].copy_(dummy_bp.kv_page_indices[:n_pages])

        self.flashinferWrapper.prepareAttention(
            'decode',
            self.batch_position,
            kvCachePool.page_len,
            "NONE",
            kvCachePool.cache_data[0].dtype
        )

        # ── Capture CUDA Graph ────────────────────────────────────
        dummy_tok = torch.zeros((B, L), dtype=torch.long, device=device)
        dummy_pos = torch.zeros_like(dummy_tok)
        self.input_ids_buf.copy_(dummy_tok)
        self.position_ids_buf.copy_(dummy_pos)

        # Warmup
        for _ in range(3):
            _ = self(
                self.input_ids_buf,
                with_softmax=True,
                position_ids=self.position_ids_buf,
                kvCachePool=kvCachePool,
                batch_position=self.batch_position,
                mode="decode", 
                flashinferWrapper=self.flashinferWrapper,
            )
        
        torch.cuda.synchronize()

        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device=device))
        
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg, stream=capture_stream):
            self.output_buffer = self(
                self.input_ids_buf,
                with_softmax=True,
                position_ids=self.position_ids_buf,
                kvCachePool=kvCachePool,
                batch_position=self.batch_position,
                mode="decode", 
                flashinferWrapper=self.flashinferWrapper,
            )
        
        self.graph = cg
        torch.cuda.current_stream(device=device).wait_stream(capture_stream)
        
        dummy_request.release()
        print("Finished capturing draft model CUDA graph for Sequence Decoding", flush=True)

    def set_important_head_idx(self, filename, generator_kwargs):
        with open(filename, 'r') as f:
            important_heads = json.load(f)
        self.important_heads = [important_heads[str(i)] for i in range(len(important_heads))]
        self.important_heads = torch.tensor(self.important_heads, device=self.device)[:,:generator_kwargs['SRH_head_num']]
        self.important_heads = self.important_heads[self.important_layers.to("cpu")]
        print("self.important_heads", self.important_heads)

    def set_important_layers(self, layer_budget_filename, list_srh_filename, generator_kwargs, dataset_name="avg_score"):
        if generator_kwargs['SRH_select_method'] == "layer_budget":
            # layer budget based selection
            with open(layer_budget_filename, 'r') as f:
                important_layers_scores = json.load(f)[dataset_name]
            scores_per_layer = torch.tensor(important_layers_scores, device=self.device)
        else:
            # SRH score aggregation based selection
            # select top-k SRH head then aggregate their scores to select layers
            with open(list_srh_filename, 'r') as f:
                head_scores = json.load(f)
                
            # get the top-k heads in each layer and aggregate their scores
            SRH_score_per_layer = []
            for layer_idx in range(len(head_scores)):
                heads_scores = head_scores[str(layer_idx)]
                topk_heads_scores = torch.topk(torch.tensor(heads_scores, device=self.device), k=generator_kwargs['SRH_head_num']).values
                SRH_score_per_layer.append(torch.sum(topk_heads_scores).item())
            scores_per_layer = torch.tensor(SRH_score_per_layer, device=self.device)
        
        self.important_layers = torch.sort(torch.topk(scores_per_layer, k=generator_kwargs['SRH_layer_num']).indices).values
        print("self.important_layers", self.important_layers)

    def decode_step(
        self, 
        token_ids: torch.Tensor, 
        position_ids: torch.Tensor, 
        batch_position: KvCacheBatchPosition,
        kvCachePool: KvCachePool
    ):
        """
        Executes a single decode step using CUDA Graph.
        Updates static buffers -> Prepares Attention (on static buffers) -> Replays Graph.
        """
        L = token_ids.shape[1]
        
        # ---- buffer updates ------------------------------------------------
        self.input_ids_buf[:, :L].copy_(token_ids)
        self.position_ids_buf[:, :L].copy_(position_ids)

        self.seq_indptr_buf.copy_(batch_position.seq_indptr)
        self.kv_page_indptr_buf.copy_(batch_position.kv_page_indptr)
        self.kv_last_page_len_buf.copy_(batch_position.kv_last_page_len)
        self.batch_indices_buf.copy_(batch_position.batch_indices)
        self.positions_buf.copy_(batch_position.positions)

        n_pages = batch_position.kv_page_indptr[1].item()
        self.kv_page_indices_buf[:n_pages].copy_(batch_position.kv_page_indices[:n_pages])

        # Prepare Attention using the STATIC buffer wrapper (self.batch_position)
        self.flashinferWrapper.prepareAttention(
            'decode', 
            self.batch_position,
            kvCachePool.page_len,
            "NONE",
            kvCachePool.cache_data[0].dtype,
        )

        # ---- replay --------------------------------------------------------
        self.graph.replay()
        return self.output_buffer

    @torch.no_grad()
    def speculate(self, input_ids, request_kv_cache, **kwargs):
        # 1) Obtain necessary parameters
        device = input_ids.device
        batch_size, input_len = input_ids.shape
        self.request_kv_cache = request_kv_cache
        
        if not hasattr(self, 'flashinferWrapper'):
            self.flashinferWrapper = FlashinferAttentionWrapper(
                self.model.config.num_attention_heads, 
                self.model.config.num_key_value_heads, 
                self.model.config.hidden_size, 
                request_kv_cache.kvCachePool.page_len
            )
        
        self.kvCachePool = request_kv_cache.kvCachePool
        assert batch_size == 1, "Only support batch_size=1 for now."

        draft_ids = [input_ids[:, -1:]] 
        
        # 2) Initialize kv_len
        with nvtx.annotate("Initialize kv_len"):
            kv_len = request_kv_cache.get_seq_length()
            if isinstance(kv_len, torch.Tensor):
                kv_len = kv_len.item()

        # 3) First forward pass (Prefill)
        new_tokens = input_ids[:, kv_len:]
        num_new = new_tokens.shape[1]
        current_pos_ids = torch.arange(kv_len, kv_len + num_new, dtype=torch.long, device=device).unsqueeze(0)

        with nvtx.annotate("draft prefill", color="red"):
            request_kv_cache.increment(num_new)
            batch_position = getKvCacheBatchPosition(
                request_kv_caches=[request_kv_cache],
                mode='tree', 
                device=device,
                treeTokens=num_new,
            )
            self.flashinferWrapper.prepareAttention(
                'prefill',
                batch_position,
                self.kvCachePool.page_len,
                "NONE", 
                self.kvCachePool.cache_data[0].dtype,
            )
            logits = self(
                new_tokens,
                with_softmax=True,
                logits_to_keep=1, 
                position_ids=current_pos_ids,
                kvCachePool=request_kv_cache.kvCachePool,
                batch_position=batch_position,
                mode='prefill',
                flashinferWrapper=self.flashinferWrapper,
            )
            kv_len += num_new

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        draft_ids.append(next_token)

        # 4) Autoregressive Loop - INSIDE context manager
        # WRAPPER: Use the Context Manager here to activate the monkey patch
        with CaptureAttentionContext(self):
            for i in range(self.draft_params.max_depth - 1):
                with nvtx.annotate("draft step", color="green"):
                    request_kv_cache.increment(1)
                    
                    current_input = draft_ids[-1]
                    if current_input.dim() > 2:
                        current_input = current_input.view(batch_size, 1)

                    current_pos_ids = torch.tensor([[kv_len]], device=device, dtype=torch.long)
                    
                    # Get new batch position data
                    batch_position = getKvCacheBatchPosition(
                        request_kv_caches=[request_kv_cache],
                        mode='decode', 
                        device=device,
                        treeTokens=1,
                    )
                    
                    # If graph captured, use decode_step
                    if hasattr(self, "graph"):
                        logits = self.decode_step(
                            current_input,
                            current_pos_ids,
                            batch_position,
                            self.kvCachePool
                        )
                    else:
                        self.flashinferWrapper.prepareAttention(
                            'decode', 
                            batch_position,
                            request_kv_cache.kvCachePool.page_len,
                            "NONE",
                            request_kv_cache.kvCachePool.cache_data[0].dtype,
                        )
                        logits = self(
                            current_input,
                            with_softmax=True,
                            position_ids=current_pos_ids,
                            kvCachePool=request_kv_cache.kvCachePool,
                            batch_position=batch_position,
                            mode='decode',
                            flashinferWrapper=self.flashinferWrapper,
                        )
                    
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    draft_ids.append(next_token)
                    kv_len += 1
        
        print("self.latest_captured_rope_queries.shape:", self.latest_captured_rope_queries.shape)

        return torch.cat(draft_ids, dim=-1)