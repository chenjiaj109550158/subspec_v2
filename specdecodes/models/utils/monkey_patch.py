import torch
import types
from typing import Optional, Union, TYPE_CHECKING

# ==============================================================================
# FlashInfer-Instance-Safe Monkey Patch
# Designed to capture RoPE queries even when FlashInfer binds methods to instances.
# ==============================================================================

if TYPE_CHECKING:
    class TargetKVSDDraftModel:
        latest_captured_queries: Union[list, torch.Tensor]
        latest_captured_rope_queries: Union[list, torch.Tensor]
        _capture_enabled: bool
        important_layers: torch.Tensor
        important_heads: torch.Tensor
        model: torch.nn.Module # Assuming HF structure
        def __init__(self): pass

class CaptureAttentionContext:
    """
    Instance-level monkey patch.
    It iterates over the draft model's layers and patches the bound 'forward' method
    of each self_attn module. This bypasses the issue where FlashInfer overrides class methods.
    """

    def __init__(self, draft_model: "TargetKVSDDraftModel"):
        self.draft_model = draft_model
        self.original_methods = {} # Store original bound methods: {module_instance: original_method}
        self.layer_to_idx = {}
        
        # We need access to the underlying transformers model to find layers
        # Usually it's draft_model.model
        if hasattr(draft_model, "model"):
            self.base_model = draft_model.model
        else:
            self.base_model = draft_model

    def __enter__(self):
        dm = self.draft_model
        dm.latest_captured_queries = []
        dm.latest_captured_rope_queries = []
        dm._capture_enabled = True

        # 1. Map Layer Indices for filtering
        if hasattr(dm, 'important_layers') and dm.important_layers is not None:
            imp_layers_list = dm.important_layers.tolist() if torch.is_tensor(dm.important_layers) else dm.important_layers
            self.layer_to_idx = {layer_idx: i for i, layer_idx in enumerate(imp_layers_list)}
        else:
            self.layer_to_idx = {}

        ctx = self

        # 2. Define the Patch Creator
        def make_patched_forward(original_forward_method):
            """
            Creates a wrapper around the specific instance's forward method.
            """
            def patched_forward(self_attn, hidden_states, *args, **kwargs):
                # --- [Shadow Computation: Capture RoPE] ---
                try:
                    capture_enabled = getattr(dm, "_capture_enabled", False)
                    # Use self_attn.layer_idx. usually available in HF models
                    layer_idx = getattr(self_attn, "layer_idx", -1)
                    is_important_layer = layer_idx in ctx.layer_to_idx
                    
                    if capture_enabled and is_important_layer:
                        # Reconstruct shapes
                        input_shape = hidden_states.shape[:-1]
                        hidden_shape = (*input_shape, -1, self_attn.head_dim)
                        
                        # A. Shadow Compute Q (Pre-RoPE)
                        # We use the q_proj weights directly
                        query_states = self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                        
                        # Filter & Store Q
                        imp_idx = ctx.layer_to_idx.get(layer_idx)
                        head_indices = dm.important_heads[imp_idx]
                        if head_indices.device != query_states.device:
                            head_indices = head_indices.to(query_states.device)
                            
                        dm.latest_captured_queries.append(
                            query_states.index_select(1, head_indices).detach().clone()
                        )

                        # B. Shadow Compute RoPE
                        # We need to manually apply RoPE using the standard Llama/Transformers logic
                        # independent of what FlashInfer does internally.
                        
                        position_ids = kwargs.get('position_ids', None)
                        if position_ids is None and len(args) > 1:
                             # Try to guess position_ids from args if positional
                             # standard forward: (hidden_states, attention_mask, position_ids, ...)
                             # But let's rely on kwargs or try to inspect
                             pass

                        if position_ids is not None and hasattr(self_attn, 'rotary_emb'):
                            cos, sin = self_attn.rotary_emb(query_states, position_ids)
                            
                            # We need a standard apply_rope function. 
                            # Since we can't rely on the patched module's apply_rope (it might be gone or changed),
                            # we implement a minimal one or import it.
                            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                            
                            # Apply RoPE (Shadow)
                            q_rope, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin, position_ids)
                            
                            # Filter & Store RoPE Q
                            dm.latest_captured_rope_queries.append(
                                q_rope.index_select(1, head_indices).detach().clone()
                            )

                except Exception:
                    # Capture failed, proceed to run model anyway
                    pass
                # --- [End Shadow Computation] ---

                # 3. Call the ORIGINAL FlashInfer forward
                return original_forward_method(hidden_states, *args, **kwargs)
            
            return patched_forward

        # 3. Iterate Layers and Apply Patch to Instances
        # Assuming standard HF Llama structure: model.layers[i].self_attn
        if hasattr(self.base_model, "layers"):
            layers = self.base_model.layers
        elif hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            layers = self.base_model.model.layers
        else:
            return # Cannot find layers

        for i, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                module = layer.self_attn
                
                # Check if this module is already patched or has a bound method
                if "forward" in module.__dict__:
                    # FlashInfer (or others) has bound a method to this instance
                    original_bound_method = module.__dict__["forward"]
                    
                    # Save original
                    self.original_methods[module] = original_bound_method
                    
                    # Create patched version binding it to the instance
                    # We use types.MethodType to bind the function as a method to the instance
                    patched = make_patched_forward(original_bound_method)
                    
                    # Overwrite the instance method
                    # Note: make_patched_forward returns a function that takes (self_attn, ...).
                    # When we assign it to __dict__['forward'], we usually expect a bound method behavior 
                    # OR a raw function if we are replacing what _bind_method_to_module did.
                    # _bind_method_to_module did: module.__dict__[name] = method.__get__(module) (which is a bound method)
                    # So we should likely store a bound method or a callable that acts like one.
                    
                    # Wrapper to handle 'self' correctly
                    # The original_bound_method is already bound to 'module'.
                    # Our patched_forward expects 'self_attn' as first arg.
                    
                    # Let's bind our patched_forward to the module
                    bound_patched = types.MethodType(patched, module)
                    module.forward = bound_patched # This updates __dict__ implicitly or via descriptor
                    
                else:
                    # If no instance method, it uses the Class method.
                    # We can still force an instance method to override it.
                    original_class_method = getattr(module.__class__, "forward")
                    
                    # We don't save to self.original_methods here because we just delete the instance attr to restore
                    self.original_methods[module] = "CLASS_LEVEL"
                    
                    patched = make_patched_forward(original_class_method)
                    bound_patched = types.MethodType(patched, module)
                    module.forward = bound_patched

        return dm

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original methods
        for module, original in self.original_methods.items():
            if original == "CLASS_LEVEL":
                # Delete the instance method so it falls back to class
                del module.forward
            else:
                # Restore the original bound method
                module.forward = original
        
        self.draft_model._capture_enabled = False
        
        # ... (Stacking logic remains the same as before) ...
        num_layers = len(self.draft_model.important_layers) if hasattr(self.draft_model, 'important_layers') else 0
        if num_layers == 0: return 

        def stack_and_reshape(captured_list):
            if not captured_list: return None
            processed_list = []
            for q in captured_list:
                if q.dim() == 4 and q.size(2) > 1:
                    q = q[:, :, -1:, :]
                processed_list.append(q)
            try:
                stacked = torch.stack(processed_list)
            except Exception: return None
            total_items = stacked.shape[0]
            if num_layers > 0 and total_items % num_layers != 0: return stacked
            num_steps = total_items // num_layers
            reshaped = stacked.view(num_steps, num_layers, *stacked.shape[1:])
            if reshaped.shape[-2] == 1: reshaped = reshaped.squeeze(-2)
            return reshaped

        self.draft_model.latest_captured_queries = stack_and_reshape(self.draft_model.latest_captured_queries)
        self.draft_model.latest_captured_rope_queries = stack_and_reshape(self.draft_model.latest_captured_rope_queries)