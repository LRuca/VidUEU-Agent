# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any
import os

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from data.citation_dialogue_utils import (
    GEN_CITATION_SYSTEM_PROMPT,
    VLM_CITATION_SYSTEM_PROMPT,
    classify_assistant_round,
    collect_first_appearance_images_for_text,
    extract_turn_image_maps,
    format_chatml_assistant_prefix,
    format_chatml_utterance,
    merge_image_uri_map,
    remove_citation_markup_from_history_text,
    resolve_turn_id,
)
from modeling.bagel.qwen2_navit import NaiveCache



VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''

class InterleaveInferencer:
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids
        
    def init_gen_context(self): 
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        # used for interleave data, currently only support 1 data inference, 

        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)        
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        # used for interleave data, currently only support 1 data inference, 

        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes =  gen_context['ropes']

        if vae:
            ## update vae
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            ## update vit
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_latent(self, latent, image_shape, gen_context):
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input, kv_lens, ropes = self.model.prepare_vae_latents(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            packed_latents=[latent],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        past_key_values = self.model.forward_cache_update_vae_latents(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        
        num_timesteps=50, 
        timestep_shift=3.0,
        enable_taylorseer=False,
        return_latent=False,
    ):
        # print(cfg_renorm_type)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # text cfg
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # img cfg
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        if return_latent:
            return image, unpacked_latent[0].detach(), image_shape
        return image

        
    def decode_image(self, latent, image_shape):
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        image = self.vae_model.decode(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output
        
    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
        citation=False,
        return_latent=False,
    ) -> List[Union[str, Image.Image]]:

        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think and citation:
                raise ValueError("think and citation cannot both be True.")

            if think or citation:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT if think else VLM_CITATION_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT if think else GEN_CITATION_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                elif isinstance(input_term, dict) and input_term.get("type") == "vae_latent":
                    latent = input_term["latent"]
                    latent_shape = tuple(input_term["image_shape"])
                    gen_context = self.update_context_latent(latent, latent_shape, gen_context)
                    vit_image = input_term.get("vit_image")
                    if vit_image is not None:
                        vit_image = self.vae_transform.resize_transform(pil_img2rgb(vit_image))
                        gen_context = self.update_context_image(vit_image, gen_context, vae=False, vit=True)
                    image_shapes = latent_shape
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                if think or citation:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                    return_latent=return_latent,
                )

                if return_latent:
                    img, latent, latent_shape = img
                    output_list.append(img)
                    output_list.append({"type": "vae_latent", "latent": latent, "image_shape": latent_shape})
                else:
                    output_list.append(img)

        return output_list

    @staticmethod
    def build_citation_input_list_from_dialogue(
        dialogue: Dict[str, Any],
        target_turn_index: int,
        include_turn_in_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Build inference input_list from WOOL/WEAVE-style dialogue:
        - include full history + current-turn user query
        - insert image only at its first appearance position in this sample
        - history assistant text strips <cite>/<claim> markup
        - context keeps role prefixes; turn prefix is optional by flag
        """
        turns = dialogue.get("turns", []) if isinstance(dialogue, dict) else []
        if not isinstance(turns, list) or len(turns) == 0:
            raise ValueError("dialogue.turns must be a non-empty list.")
        if target_turn_index < 0 or target_turn_index >= len(turns):
            raise ValueError(f"target_turn_index out of range: {target_turn_index}")

        curr_turn = turns[target_turn_index]
        curr_assistant = curr_turn.get("assistant", {}) or {}
        round_type = classify_assistant_round(curr_assistant)
        if round_type not in {"text", "image"}:
            raise ValueError(f"unsupported assistant round type at turn={target_turn_index + 1}")

        known_image_uri_map: Dict[int, str] = {}
        inserted_image_nums = set()
        input_list: List[Union[str, Image.Image]] = []

        def _append_text_piece(piece: str):
            if not isinstance(piece, str) or len(piece) == 0:
                return
            if len(input_list) > 0 and isinstance(input_list[-1], str):
                input_list[-1] = input_list[-1] + piece
            else:
                input_list.append(piece)

        def _append_image(uri: str):
            if not isinstance(uri, str) or len(uri) == 0:
                return
            if not os.path.exists(uri):
                raise FileNotFoundError(f"Image not found: {uri}")
            img = Image.open(uri)
            img = pil_img2rgb(img)
            try:
                setattr(img, "_bagel_source_uri", uri)
            except Exception:
                pass
            input_list.append(img)

        def _append_utterance(turn_no: int, role: str, text: str):
            text = text if isinstance(text, str) else ""
            first_images = collect_first_appearance_images_for_text(
                text=text,
                image_uri_map=known_image_uri_map,
                inserted_image_nums=inserted_image_nums,
            )
            for it in first_images:
                uri = it.get("uri", "")
                if isinstance(uri, str) and uri:
                    _append_image(uri)

            utterance = format_chatml_utterance(
                role=role,
                text=text,
                include_turn=include_turn_in_context,
                turn_no=turn_no,
            )
            _append_text_piece(utterance)

        for idx in range(target_turn_index + 1):
            turn = turns[idx]
            if not isinstance(turn, dict):
                continue
            turn_no = resolve_turn_id(turn, idx)
            user = turn.get("user", {}) or {}
            assistant = turn.get("assistant", {}) or {}

            user_map, assistant_map = extract_turn_image_maps(turn)
            merge_image_uri_map(known_image_uri_map, user_map)
            _append_utterance(turn_no, "user", user.get("text", "") or "")

            if idx == target_turn_index:
                continue

            merge_image_uri_map(known_image_uri_map, assistant_map)
            hist_text = remove_citation_markup_from_history_text(assistant.get("text", "") or "")
            if (not hist_text) and len(assistant_map) > 0:
                first_num = list(assistant_map.keys())[0]
                hist_text = f"Image #{first_num}"
            _append_utterance(turn_no, "assistant", hist_text)

        _append_text_piece(format_chatml_assistant_prefix())

        return {
            "input_list": input_list,
            "understanding_output": (round_type == "text"),
            "round_type": round_type,
        }
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
