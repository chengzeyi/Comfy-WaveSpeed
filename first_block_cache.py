import contextlib
import dataclasses
import unittest
from collections import defaultdict
from typing import DefaultDict, Dict

import torch


@dataclasses.dataclass
class CacheContext:
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int))

    def get_incremental_name(self, name=None):
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

    @torch.compiler.disable()
    def get_buffer(self, name):
        return self.buffers.get(name)

    @torch.compiler.disable()
    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()


@torch.compiler.disable()
def get_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable()
def set_buffer(name, buffer):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


_current_cache_context = None


def create_cache_context():
    return CacheContext()


def get_current_cache_context():
    return _current_cache_context


def set_current_cache_context(cache_context=None):
    global _current_cache_context
    _current_cache_context = cache_context


@contextlib.contextmanager
def cache_context(cache_context):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


def patch_get_output_data():
    import execution

    get_output_data = getattr(execution, "get_output_data", None)
    if get_output_data is None:
        return

    if getattr(get_output_data, "_patched", False):
        return

    def new_get_output_data(*args, **kwargs):
        out = get_output_data(*args, **kwargs)
        cache_context = get_current_cache_context()
        if cache_context is not None:
            cache_context.clear_buffers()
            set_current_cache_context(None)
        return out

    new_get_output_data._patched = True
    execution.get_output_data = new_get_output_data


@torch.compiler.disable()
def are_two_tensors_similar(t1, t2, *, threshold):
    if t1.shape != t2.shape:
        return False
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    diff = mean_diff / mean_t1
    return diff.item() < threshold


@torch.compiler.disable()
def apply_prev_hidden_states_residual(hidden_states,
                                      encoder_hidden_states=None):
    hidden_states_residual = get_buffer("hidden_states_residual")
    assert hidden_states_residual is not None, "hidden_states_residual must be set before"
    hidden_states = hidden_states_residual + hidden_states
    hidden_states = hidden_states.contiguous()

    if encoder_hidden_states is None:
        return hidden_states

    encoder_hidden_states_residual = get_buffer(
        "encoder_hidden_states_residual")
    if encoder_hidden_states_residual is None:
        encoder_hidden_states = None
    else:
        encoder_hidden_states = encoder_hidden_states_residual + encoder_hidden_states
        encoder_hidden_states = encoder_hidden_states.contiguous()

    return hidden_states, encoder_hidden_states


@torch.compiler.disable()
def get_can_use_cache(first_hidden_states_residual,
                      threshold,
                      parallelized=False):
    prev_first_hidden_states_residual = get_buffer(
        "first_hidden_states_residual")
    can_use_cache = prev_first_hidden_states_residual is not None and are_two_tensors_similar(
        prev_first_hidden_states_residual,
        first_hidden_states_residual,
        threshold=threshold,
    )
    return can_use_cache


class CachedTransformerBlocks(torch.nn.Module):

    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        residual_diff_threshold,
        validate_can_use_cache_function=None,
        return_hidden_states_first=True,
        accept_hidden_states_first=True,
        cat_hidden_states_first=False,
        return_hidden_states_only=False,
        clone_original_hidden_states=False,
    ):
        super().__init__()
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.validate_can_use_cache_function = validate_can_use_cache_function
        self.return_hidden_states_first = return_hidden_states_first
        self.accept_hidden_states_first = accept_hidden_states_first
        self.cat_hidden_states_first = cat_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        self.clone_original_hidden_states = clone_original_hidden_states

    def forward(self, *args, **kwargs):
        img_arg_name = None
        if "img" in kwargs:
            img_arg_name = "img"
        elif "hidden_states" in kwargs:
            img_arg_name = "hidden_states"
        txt_arg_name = None
        if "txt" in kwargs:
            txt_arg_name = "txt"
        elif "context" in kwargs:
            txt_arg_name = "context"
        elif "encoder_hidden_states" in kwargs:
            txt_arg_name = "encoder_hidden_states"
        if self.accept_hidden_states_first:
            if args:
                img = args[0]
                args = args[1:]
            else:
                img = kwargs.pop(img_arg_name)
            if args:
                txt = args[0]
                args = args[1:]
            else:
                txt = kwargs.pop(txt_arg_name)
        else:
            if args:
                txt = args[0]
                args = args[1:]
            else:
                txt = kwargs.pop(txt_arg_name)
            if args:
                img = args[0]
                args = args[1:]
            else:
                img = kwargs.pop(img_arg_name)
        hidden_states = img
        encoder_hidden_states = txt
        if self.residual_diff_threshold <= 0.0:
            for block in self.transformer_blocks:
                if txt_arg_name == "encoder_hidden_states":
                    hidden_states = block(
                        hidden_states,
                        *args,
                        encoder_hidden_states=encoder_hidden_states,
                        **kwargs)
                else:
                    if self.accept_hidden_states_first:
                        hidden_states = block(hidden_states,
                                              encoder_hidden_states, *args,
                                              **kwargs)
                    else:
                        hidden_states = block(encoder_hidden_states,
                                              hidden_states, *args, **kwargs)
                if not self.return_hidden_states_only:
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.return_hidden_states_first:
                        hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat(
                    [hidden_states, encoder_hidden_states]
                    if self.cat_hidden_states_first else
                    [encoder_hidden_states, hidden_states],
                    dim=1)
                for block in self.single_transformer_blocks:
                    hidden_states = block(hidden_states, *args, **kwargs)
                hidden_states = hidden_states[:,
                                              encoder_hidden_states.shape[1]:]
            if self.return_hidden_states_only:
                return hidden_states
            else:
                return ((hidden_states, encoder_hidden_states)
                        if self.return_hidden_states_first else
                        (encoder_hidden_states, hidden_states))

        original_hidden_states = hidden_states
        if self.clone_original_hidden_states:
            original_hidden_states = original_hidden_states.clone()
        first_transformer_block = self.transformer_blocks[0]
        if txt_arg_name == "encoder_hidden_states":
            hidden_states = first_transformer_block(
                hidden_states,
                *args,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs)
        else:
            if self.accept_hidden_states_first:
                hidden_states = first_transformer_block(
                    hidden_states, encoder_hidden_states, *args, **kwargs)
            else:
                hidden_states = first_transformer_block(
                    encoder_hidden_states, hidden_states, *args, **kwargs)
        if not self.return_hidden_states_only:
            hidden_states, encoder_hidden_states = hidden_states
            if not self.return_hidden_states_first:
                hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        can_use_cache = get_can_use_cache(
            first_hidden_states_residual,
            threshold=self.residual_diff_threshold,
        )
        if self.validate_can_use_cache_function is not None:
            can_use_cache = self.validate_can_use_cache_function(can_use_cache)

        torch._dynamo.graph_break()
        if can_use_cache:
            del first_hidden_states_residual
            hidden_states, encoder_hidden_states = apply_prev_hidden_states_residual(
                hidden_states, encoder_hidden_states)
        else:
            set_buffer("first_hidden_states_residual",
                       first_hidden_states_residual)
            del first_hidden_states_residual
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                encoder_hidden_states_residual,
            ) = self.call_remaining_transformer_blocks(
                hidden_states,
                encoder_hidden_states,
                *args,
                txt_arg_name=txt_arg_name,
                **kwargs)
            set_buffer("hidden_states_residual", hidden_states_residual)
            if encoder_hidden_states_residual is not None:
                set_buffer("encoder_hidden_states_residual",
                           encoder_hidden_states_residual)
        torch._dynamo.graph_break()

        if self.return_hidden_states_only:
            return hidden_states
        else:
            return ((hidden_states, encoder_hidden_states)
                    if self.return_hidden_states_first else
                    (encoder_hidden_states, hidden_states))

    def call_remaining_transformer_blocks(self,
                                          hidden_states,
                                          encoder_hidden_states,
                                          *args,
                                          txt_arg_name=None,
                                          **kwargs):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        if self.clone_original_hidden_states:
            original_hidden_states = original_hidden_states.clone()
            original_encoder_hidden_states = original_encoder_hidden_states.clone(
            )
        for block in self.transformer_blocks[1:]:
            if txt_arg_name == "encoder_hidden_states":
                hidden_states = block(
                    hidden_states,
                    *args,
                    encoder_hidden_states=encoder_hidden_states,
                    **kwargs)
            else:
                if self.accept_hidden_states_first:
                    hidden_states = block(hidden_states, encoder_hidden_states,
                                          *args, **kwargs)
                else:
                    hidden_states = block(encoder_hidden_states, hidden_states,
                                          *args, **kwargs)
            if not self.return_hidden_states_only:
                hidden_states, encoder_hidden_states = hidden_states
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        if self.single_transformer_blocks is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states]
                                      if self.cat_hidden_states_first else
                                      [encoder_hidden_states, hidden_states],
                                      dim=1)
            for block in self.single_transformer_blocks:
                hidden_states = block(hidden_states, *args, **kwargs)
            if self.cat_hidden_states_first:
                hidden_states, encoder_hidden_states = hidden_states.split(
                    [
                        hidden_states.shape[1] -
                        encoder_hidden_states.shape[1],
                        encoder_hidden_states.shape[1]
                    ],
                    dim=1)
            else:
                encoder_hidden_states, hidden_states = hidden_states.split(
                    [
                        encoder_hidden_states.shape[1],
                        hidden_states.shape[1] - encoder_hidden_states.shape[1]
                    ],
                    dim=1)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.flatten().contiguous().reshape(
            hidden_states_shape)

        if encoder_hidden_states is not None:
            encoder_hidden_states_shape = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.flatten().contiguous(
            ).reshape(encoder_hidden_states_shape)

        hidden_states_residual = hidden_states - original_hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states_residual = None
        else:
            encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states
        return hidden_states, encoder_hidden_states, hidden_states_residual, encoder_hidden_states_residual


# Based on 90f349f93df3083a507854d7fc7c3e1bb9014e24
def create_patch_unet_model__forward(model,
                                     *,
                                     residual_diff_threshold,
                                     validate_can_use_cache_function=None):
    from comfy.ldm.modules.diffusionmodules.openaimodel import timestep_embedding, forward_timestep_embed, apply_control

    def call_remaining_blocks(self, transformer_options, control,
                              transformer_patches, hs, h, *args, **kwargs):
        original_hidden_states = h

        for id, module in enumerate(self.input_blocks):
            if id < 2:
                continue
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, *args, **kwargs)
            h = apply_control(h, control, 'input')
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed(self.middle_block, h, *args, **kwargs)
        h = apply_control(h, control, 'middle')

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')

            if "output_block_patch" in transformer_patches:
                patch = transformer_patches["output_block_patch"]
                for p in patch:
                    h, hsp = p(h, hsp, transformer_options)

            h = torch.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, *args, output_shape,
                                       **kwargs)
        hidden_states_residual = h - original_hidden_states
        return h, hidden_states_residual

    def unet_model__forward(self,
                            x,
                            timesteps=None,
                            context=None,
                            y=None,
                            control=None,
                            transformer_options={},
                            **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["transformer_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames",
                                      self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", None)
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps,
                                   self.model_channels,
                                   repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        if "emb_patch" in transformer_patches:
            patch = transformer_patches["emb_patch"]
            for p in patch:
                emb = p(emb, self.model_channels, transformer_options)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        can_use_cache = False

        h = x
        for id, module in enumerate(self.input_blocks):
            if id >= 2:
                break
            transformer_options["block"] = ("input", id)
            if id == 1:
                original_h = h
            h = forward_timestep_embed(
                module,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator)
            h = apply_control(h, control, 'input')
            if "input_block_patch" in transformer_patches:
                patch = transformer_patches["input_block_patch"]
                for p in patch:
                    h = p(h, transformer_options)

            hs.append(h)
            if "input_block_patch_after_skip" in transformer_patches:
                patch = transformer_patches["input_block_patch_after_skip"]
                for p in patch:
                    h = p(h, transformer_options)

            if id == 1:
                first_hidden_states_residual = h - original_h
                can_use_cache = get_can_use_cache(
                    first_hidden_states_residual,
                    threshold=residual_diff_threshold,
                )
                if validate_can_use_cache_function is not None:
                    can_use_cache = validate_can_use_cache_function(
                        can_use_cache)
                if not can_use_cache:
                    set_buffer("first_hidden_states_residual",
                               first_hidden_states_residual)
                del first_hidden_states_residual

        torch._dynamo.graph_break()
        if can_use_cache:
            h = apply_prev_hidden_states_residual(h)
        else:
            h, hidden_states_residual = call_remaining_blocks(
                self,
                transformer_options,
                control,
                transformer_patches,
                hs,
                h,
                emb,
                context,
                transformer_options,
                time_context=time_context,
                num_video_frames=num_video_frames,
                image_only_indicator=image_only_indicator)
            set_buffer("hidden_states_residual", hidden_states_residual)
        torch._dynamo.graph_break()

        h = h.type(x.dtype)

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)

    new__forward = unet_model__forward.__get__(model)

    @contextlib.contextmanager
    def patch__forward():
        with unittest.mock.patch.object(model, "_forward", new__forward):
            yield

    return patch__forward


# Based on 90f349f93df3083a507854d7fc7c3e1bb9014e24
def create_patch_flux_forward_orig(model,
                                   *,
                                   residual_diff_threshold,
                                   validate_can_use_cache_function=None):
    from torch import Tensor
    from comfy.ldm.flux.model import timestep_embedding

    def call_remaining_blocks(self, blocks_replace, control, img, txt, vec, pe,
                              attn_mask, ca_idx, timesteps, transformer_options):
        original_hidden_states = img

        extra_block_forward_kwargs = {}
        if attn_mask is not None:
            extra_block_forward_kwargs["attn_mask"] = attn_mask

        for i, block in enumerate(self.double_blocks):
            if i < 1:
                continue
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"],
                        txt=args["txt"],
                        vec=args["vec"],
                        pe=args["pe"],
                        **extra_block_forward_kwargs)
                    return out

                out = blocks_replace[("double_block",
                                      i)]({
                                          "img": img,
                                          "txt": txt,
                                          "vec": vec,
                                          "pe": pe,
                                          **extra_block_forward_kwargs
                                      }, {
                                          "original_block": block_wrap,
                                          "transformer_options": transformer_options
                                      })
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 **extra_block_forward_kwargs)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

            # PuLID attention
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    # Will calculate influence of all pulid nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps)
                                     & (timesteps >= node_data['sigma_end'])):
                            img = img + node_data['weight'] * self.pulid_ca[
                                ca_idx](node_data['embedding'], img)
                    ca_idx += 1

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                       vec=args["vec"],
                                       pe=args["pe"],
                                       **extra_block_forward_kwargs)
                    return out

                out = blocks_replace[("single_block",
                                      i)]({
                                          "img": img,
                                          "vec": vec,
                                          "pe": pe,
                                          **extra_block_forward_kwargs
                                      }, {
                                          "original_block": block_wrap,
                                          "transformer_options": transformer_options
                                      })
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, **extra_block_forward_kwargs)

            if control is not None:  # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:, ...] += add

            # PuLID attention
            if getattr(self, "pulid_data", {}):
                real_img, txt = img[:, txt.shape[1]:,
                                    ...], img[:, :txt.shape[1], ...]
                if i % self.pulid_single_interval == 0:
                    # Will calculate influence of all nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps)
                                     & (timesteps >= node_data['sigma_end'])):
                            real_img = real_img + node_data[
                                'weight'] * self.pulid_ca[ca_idx](
                                    node_data['embedding'], real_img)
                    ca_idx += 1
                img = torch.cat((txt, real_img), 1)

        img = img[:, txt.shape[1]:, ...]

        img = img.contiguous()
        hidden_states_residual = img - original_hidden_states
        return img, hidden_states_residual

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError(
                "Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        ca_idx = 0
        extra_block_forward_kwargs = {}
        if attn_mask is not None:
            extra_block_forward_kwargs["attn_mask"] = attn_mask
        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.double_blocks):
            if i >= 1:
                break
            if ("double_block", i) in blocks_replace:

                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        img=args["img"],
                        txt=args["txt"],
                        vec=args["vec"],
                        pe=args["pe"],
                        **extra_block_forward_kwargs)
                    return out

                out = blocks_replace[("double_block",
                                      i)]({
                                          "img": img,
                                          "txt": txt,
                                          "vec": vec,
                                          "pe": pe,
                                          **extra_block_forward_kwargs
                                      }, {
                                          "original_block": block_wrap,
                                          "transformer_options": transformer_options
                                      })
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                 txt=txt,
                                 vec=vec,
                                 pe=pe,
                                 **extra_block_forward_kwargs)

            if control is not None:  # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

            # PuLID attention
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    # Will calculate influence of all pulid nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps)
                                     & (timesteps >= node_data['sigma_end'])):
                            img = img + node_data['weight'] * self.pulid_ca[
                                ca_idx](node_data['embedding'], img)
                    ca_idx += 1

            if i == 0:
                first_hidden_states_residual = img
                can_use_cache = get_can_use_cache(
                    first_hidden_states_residual,
                    threshold=residual_diff_threshold,
                )
                if validate_can_use_cache_function is not None:
                    can_use_cache = validate_can_use_cache_function(
                        can_use_cache)
                if not can_use_cache:
                    set_buffer("first_hidden_states_residual",
                               first_hidden_states_residual)
                del first_hidden_states_residual

        torch._dynamo.graph_break()
        if can_use_cache:
            img = apply_prev_hidden_states_residual(img)
        else:
            img, hidden_states_residual = call_remaining_blocks(
                self,
                blocks_replace,
                control,
                img,
                txt,
                vec,
                pe,
                attn_mask,
                ca_idx,
                timesteps,
                transformer_options,
            )
            set_buffer("hidden_states_residual", hidden_states_residual)
        torch._dynamo.graph_break()

        img = self.final_layer(img,
                               vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    new_forward_orig = forward_orig.__get__(model)

    @contextlib.contextmanager
    def patch_forward_orig():
        with unittest.mock.patch.object(model, "forward_orig",
                                        new_forward_orig):
            yield

    return patch_forward_orig
