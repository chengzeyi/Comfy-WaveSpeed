import contextlib
import unittest
import torch

from comfy import model_management

from . import first_block_cache


class ApplyFBCacheOnModel:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "object_to_patch": (
                    "STRING",
                    {
                        "default": "diffusion_model",
                    },
                ),
                "residual_diff_threshold": (
                    "FLOAT",
                    {
                        "default":
                        0.0,
                        "min":
                        0.0,
                        "max":
                        1.0,
                        "step":
                        0.001,
                        "tooltip":
                        "Controls the tolerance for caching with lower values being more strict. Setting this to 0 disables the FBCache effect.",
                    },
                ),
                "start": (
                    "FLOAT",
                    {
                        "default":
                        0.0,
                        "step":
                        0.01,
                        "max":
                        1.0,
                        "min":
                        0.0,
                        "tooltip":
                        "Start time as a percentage of sampling where the FBCache effect can apply. Example: 0.0 would signify 0% (the beginning of sampling), 0.5 would signify 50%.",
                    },
                ),
                "end": ("FLOAT", {
                    "default":
                    1.0,
                    "step":
                    0.01,
                    "max":
                    1.0,
                    "min":
                    0.0,
                    "tooltip":
                    "End time as a percentage of sampling where the FBCache effect can apply. Example: 1.0 would signify 100% (the end of sampling), 0.5 would signify 50%.",
                }),
                "max_consecutive_cache_hits": (
                    "INT",
                    {
                        "default":
                        -1,
                        "min":
                        -1,
                        "tooltip":
                        "Allows limiting how many cached results can be used in a row. For example, setting this to 1 will mean there will be at least one full model call after each cached result. Set to 0 to disable FBCache effect, or -1 to allow unlimited consecutive cache hits.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "patch"

    CATEGORY = "wavespeed"

    def patch(
        self,
        model,
        object_to_patch,
        residual_diff_threshold,
        max_consecutive_cache_hits=-1,
        start=0.0,
        end=1.0,
    ):
        if residual_diff_threshold <= 0.0 or max_consecutive_cache_hits == 0:
            return (model, )

        first_block_cache.patch_get_output_data()

        using_validation = max_consecutive_cache_hits >= 0 or start > 0 or end < 1
        if using_validation:
            model_sampling = model.get_model_object("model_sampling")
            start_sigma, end_sigma = (float(
                model_sampling.percent_to_sigma(pct)) for pct in (start, end))
            del model_sampling

            @torch.compiler.disable()
            def validate_use_cache(use_cached):
                nonlocal consecutive_cache_hits
                use_cached = use_cached and end_sigma <= current_timestep <= start_sigma
                use_cached = use_cached and (max_consecutive_cache_hits < 0
                                             or consecutive_cache_hits
                                             < max_consecutive_cache_hits)
                consecutive_cache_hits = consecutive_cache_hits + 1 if use_cached else 0
                return use_cached
        else:
            validate_use_cache = None

        prev_timestep = None
        prev_input_state = None
        current_timestep = None
        consecutive_cache_hits = 0

        def reset_cache_state():
            # Resets the cache state and hits/time tracking variables.
            nonlocal prev_input_state, prev_timestep, consecutive_cache_hits
            prev_input_state = prev_timestep = None
            consecutive_cache_hits = 0
            first_block_cache.set_current_cache_context(
                first_block_cache.create_cache_context())

        def ensure_cache_state(model_input: torch.Tensor, timestep: float):
            # Validates the current cache state and hits/time tracking variables
            # and triggers a reset if necessary. Also updates current_timestep and
            # maintains the cache context sequence number.
            nonlocal current_timestep
            input_state = (model_input.shape, model_input.dtype, model_input.device)
            cache_context = first_block_cache.get_current_cache_context()
            # We reset when:
            need_reset = (
                # The previous timestep or input state is not set
                prev_timestep is None or
                prev_input_state is None or
                # Or dtype/device have changed
                prev_input_state[1:] != input_state[1:] or
                # Or the input state after the batch dimension has changed
                prev_input_state[0][1:] != input_state[0][1:] or
                # Or there is no cache context (in this case reset is just making a context)
                cache_context is None or
                # Or the current timestep is higher than the previous one
                timestep > prev_timestep
            )
            if need_reset:
                reset_cache_state()
            elif timestep == prev_timestep:
                # When the current timestep is the same as the previous, we assume ComfyUI has split up
                # the model evaluation into multiple chunks. In this case, we increment the sequence number.
                # Note: No need to check if cache_context is None for these branches as need_reset would be True
                #       if so.
                cache_context.sequence_num += 1
            elif timestep < prev_timestep:
                # When the timestep is less than the previous one, we can reset the context sequence number
                cache_context.sequence_num = 0
            current_timestep = timestep

        def update_cache_state(model_input: torch.Tensor, timestep: float):
            # Updates the previous timestep and input state validation variables.
            nonlocal prev_timestep, prev_input_state
            prev_timestep = timestep
            prev_input_state = (model_input.shape, model_input.dtype, model_input.device)

        model = model.clone()
        diffusion_model = model.get_model_object(object_to_patch)

        if diffusion_model.__class__.__name__ in ("UNetModel", "Flux"):

            if diffusion_model.__class__.__name__ == "UNetModel":
                create_patch_function = first_block_cache.create_patch_unet_model__forward
            elif diffusion_model.__class__.__name__ == "Flux":
                create_patch_function = first_block_cache.create_patch_flux_forward_orig
            else:
                raise ValueError(
                    f"Unsupported model {diffusion_model.__class__.__name__}")

            patch_forward = create_patch_function(
                diffusion_model,
                residual_diff_threshold=residual_diff_threshold,
                validate_can_use_cache_function=validate_use_cache,
            )

            def model_unet_function_wrapper(model_function, kwargs):
                try:
                    input = kwargs["input"]
                    timestep = kwargs["timestep"]
                    c = kwargs["c"]
                    t = timestep[0].item()

                    ensure_cache_state(input, t)

                    with patch_forward():
                        result = model_function(input, timestep, **c)
                        update_cache_state(input, t)
                        return result
                except Exception as exc:
                    reset_cache_state()
                    raise exc from None
        else:
            is_non_native_ltxv = False
            if diffusion_model.__class__.__name__ == "LTXVTransformer3D":
                is_non_native_ltxv = True
                diffusion_model = diffusion_model.transformer

            double_blocks_name = None
            single_blocks_name = None
            if hasattr(diffusion_model, "transformer_blocks"):
                double_blocks_name = "transformer_blocks"
            elif hasattr(diffusion_model, "double_blocks"):
                double_blocks_name = "double_blocks"
            elif hasattr(diffusion_model, "joint_blocks"):
                double_blocks_name = "joint_blocks"
            else:
                raise ValueError(
                    f"No double blocks found for {diffusion_model.__class__.__name__}"
                )

            if hasattr(diffusion_model, "single_blocks"):
                single_blocks_name = "single_blocks"

            if is_non_native_ltxv:
                original_create_skip_layer_mask = getattr(
                    diffusion_model, "create_skip_layer_mask", None)
                if original_create_skip_layer_mask is not None:
                    # original_double_blocks = getattr(diffusion_model,
                    #                                  double_blocks_name)

                    def new_create_skip_layer_mask(self, *args, **kwargs):
                        # with unittest.mock.patch.object(self, double_blocks_name,
                        #                                 original_double_blocks):
                        #     return original_create_skip_layer_mask(*args, **kwargs)
                        # return original_create_skip_layer_mask(*args, **kwargs)
                        raise RuntimeError(
                            "STG is not supported with FBCache yet")

                    diffusion_model.create_skip_layer_mask = new_create_skip_layer_mask.__get__(
                        diffusion_model)

            cached_transformer_blocks = torch.nn.ModuleList([
                first_block_cache.CachedTransformerBlocks(
                    None if double_blocks_name is None else getattr(
                        diffusion_model, double_blocks_name),
                    None if single_blocks_name is None else getattr(
                        diffusion_model, single_blocks_name),
                    residual_diff_threshold=residual_diff_threshold,
                    validate_can_use_cache_function=validate_use_cache,
                    cat_hidden_states_first=diffusion_model.__class__.__name__
                    == "HunyuanVideo",
                    return_hidden_states_only=diffusion_model.__class__.
                    __name__ == "LTXVModel" or is_non_native_ltxv,
                    clone_original_hidden_states=diffusion_model.__class__.
                    __name__ == "LTXVModel",
                    return_hidden_states_first=diffusion_model.__class__.
                    __name__ != "OpenAISignatureMMDITWrapper",
                    accept_hidden_states_first=diffusion_model.__class__.
                    __name__ != "OpenAISignatureMMDITWrapper",
                )
            ])
            dummy_single_transformer_blocks = torch.nn.ModuleList()

            def model_unet_function_wrapper(model_function, kwargs):
                try:
                    input = kwargs["input"]
                    timestep = kwargs["timestep"]
                    c = kwargs["c"]
                    t = timestep[0].item()

                    ensure_cache_state(input, t)

                    with unittest.mock.patch.object(
                            diffusion_model,
                            double_blocks_name,
                            cached_transformer_blocks,
                    ), unittest.mock.patch.object(
                            diffusion_model,
                            single_blocks_name,
                            dummy_single_transformer_blocks,
                    ) if single_blocks_name is not None else contextlib.nullcontext(
                    ):
                        result = model_function(input, timestep, **c)
                        update_cache_state(input, t)
                        return result
                except Exception as exc:
                    reset_cache_state()
                    raise exc from None

        model.set_model_unet_function_wrapper(model_unet_function_wrapper)
        return (model, )
