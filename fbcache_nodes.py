import contextlib
import unittest
import torch

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
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
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
    ):
        prev_timestep = None

        model = model.clone()
        diffusion_model = model.get_model_object(object_to_patch)

        double_blocks_name = None
        single_blocks_name = None
        if hasattr(diffusion_model, "transformer_blocks"):
            double_blocks_name = "transformer_blocks"
        elif hasattr(diffusion_model, "double_blocks"):
            double_blocks_name = "double_blocks"
        elif hasattr(diffusion_model, "joint_blocks"):
            double_blocks_name = "joint_blocks"
        else:
            raise ValueError("No transformer blocks found")

        if hasattr(diffusion_model, "single_blocks"):
            single_blocks_name = "single_blocks"

        cached_transformer_blocks = torch.nn.ModuleList([
            first_block_cache.CachedTransformerBlocks(
                None if double_blocks_name is None else getattr(
                    diffusion_model, double_blocks_name),
                None if single_blocks_name is None else getattr(
                    diffusion_model, single_blocks_name),
                residual_diff_threshold=residual_diff_threshold,
                cat_hidden_states_first=diffusion_model.__class__.__name__ ==
                "HunyuanVideo",
                return_hidden_states_only=diffusion_model.__class__.__name__ ==
                "LTXVModel",
                clone_original_hidden_states=diffusion_model.__class__.__name__
                == "LTXVModel",
                return_hidden_states_first=diffusion_model.__class__.__name__
                != "OpenAISignatureMMDITWrapper",
                accept_hidden_states_first=diffusion_model.__class__.__name__
                != "OpenAISignatureMMDITWrapper",
            )
        ])
        dummy_single_transformer_blocks = torch.nn.ModuleList()

        def model_unet_function_wrapper(model_function, kwargs):
            nonlocal prev_timestep

            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            t = timestep[0].item()

            if prev_timestep is None or t >= prev_timestep:
                prev_timestep = t
                first_block_cache.set_current_cache_context(
                    first_block_cache.create_cache_context())

            with unittest.mock.patch.object(
                    diffusion_model,
                    double_blocks_name,
                    cached_transformer_blocks,
            ), unittest.mock.patch.object(
                    diffusion_model,
                    single_blocks_name,
                    dummy_single_transformer_blocks,
            ) if single_blocks_name is not None else contextlib.nullcontext():
                return model_function(input, timestep, **c)

        model.set_model_unet_function_wrapper(model_unet_function_wrapper)
        return (model, )
