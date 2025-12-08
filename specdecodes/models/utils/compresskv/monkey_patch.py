import torch
from typing import Optional, Union, TYPE_CHECKING
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# TODO: Add Qwen2.5, Qwen3, maybe Mistral

if TYPE_CHECKING:
    class TargetKVSDDraftModel:
        latest_captured_queries: Union[list, torch.Tensor]
        latest_captured_rope_queries: Union[list, torch.Tensor]
        _capture_enabled: bool
        important_layers: torch.Tensor
        important_heads: torch.Tensor
        def __init__(self): pass

class CaptureAttentionContext:
    """
    Instance-level monkey patch optimized for Llama models.
    """
    def __init__(self, draft_model: "TargetKVSDDraftModel"):
        self.draft_model = draft_model
        self.original_instance_forwards = {} 
        self.layer_to_idx = {}

    def __enter__(self):
        dm = self.draft_model
        dm.latest_captured_rope_queries = []
        dm._capture_enabled = True

        # 1. Map Layer Indices
        if hasattr(dm, 'important_layers') and dm.important_layers is not None:
            imp_layers_list = dm.important_layers.tolist() if torch.is_tensor(dm.important_layers) else dm.important_layers
            self.layer_to_idx = {layer_idx: i for i, layer_idx in enumerate(imp_layers_list)}
        else:
            return self

        # 2. Locate and Patch Layers
        base_model = getattr(dm, "model", dm)
        if hasattr(base_model, "model"): 
            base_model = base_model.model
        
        layers = getattr(base_model, "layers", [])
        for layer_idx, layer in enumerate(layers):
            if layer_idx in self.layer_to_idx:
                attn_module = getattr(layer, "self_attn", None)
                if attn_module:
                    self._patch_instance(attn_module, layer_idx)
        return self

    def _patch_instance(self, module, layer_idx):
        original_forward = module.forward
        self.original_instance_forwards[module] = original_forward
        
        ctx_head_idx = self.layer_to_idx[layer_idx]
        dm = self.draft_model

        def patched_forward(hidden_states, *args, **kwargs):
            if dm._capture_enabled:
                current_heads = dm.important_heads[ctx_head_idx]
                if current_heads.device != hidden_states.device:
                    current_heads = current_heads.to(hidden_states.device)
                
                kwargs['capture_storage'] = dm.latest_captured_rope_queries
                kwargs['important_heads'] = current_heads

            return original_forward(hidden_states, *args, **kwargs)

        module.forward = patched_forward

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original forwards
        for module, orig_forward in self.original_instance_forwards.items():
            module.forward = orig_forward
        
        self.draft_model._capture_enabled = False
        self.original_instance_forwards.clear()

        # Stack and Reshape Logic
        captured = self.draft_model.latest_captured_rope_queries
        if not captured: return

        num_layers = len(self.layer_to_idx)
        stacked = torch.stack([q[:, :, -1:, :] if q.dim() == 4 and q.size(2) > 1 else q for q in captured])
        
        if num_layers > 0 and stacked.shape[0] % num_layers == 0:
            num_steps = stacked.shape[0] // num_layers
            reshaped = stacked.view(num_steps, num_layers, *stacked.shape[1:])
            if reshaped.shape[-2] == 1: reshaped = reshaped.squeeze(-2)
            self.draft_model.latest_captured_rope_queries = reshaped
        else:
            self.draft_model.latest_captured_rope_queries = stacked