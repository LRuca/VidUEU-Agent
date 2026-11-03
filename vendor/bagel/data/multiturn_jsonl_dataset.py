# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from PIL import Image

from .data_utils import pil_img2rgb


class MultiTurnJSONLIterableDataset(torch.utils.data.IterableDataset):
    """
    Read JSONL where each line is a multi-turn episode:
      {
        "id": ...,
        "turns": [ { "t": ..., "in": {...}, "gt": {...} }, ... ],
        ...
      }

    This dataset EXPANDS each episode into per-turn training samples, where each yielded sample
    conforms to PackedDataset.pack_sequence() contract:
      - image_tensor_list: list[Tensor(C,H,W)]
      - text_ids_list: list[list[int]]
      - sequence_plan: list[dict] where dict["type"] in {"text","vit_image","vae_image"}
      - num_tokens: int (upper-bound estimate used for filtering)
      - data_indexes: dict (for resuming)
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        local_rank: int,
        world_size: int,
        num_workers: int,
        data_status: Optional[Dict] = None,
        data_dir_list: Optional[List[str]] = None,
        jsonl_path_list: Optional[List[str]] = None,
        transform=None,
        vit_transform=None,
        only_current_turn_images: bool = False,
        shuffle_lines: bool = True,
        shuffle_seed: int = 0,
        num_used_data: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        if jsonl_path_list is None or len(jsonl_path_list) == 0:
            raise ValueError("jsonl_path_list is required for MultiTurnJSONLIterableDataset.")
        if data_dir_list is None or len(data_dir_list) == 0:
            raise ValueError("data_dir_list is required (used as image root) for MultiTurnJSONLIterableDataset.")
        if transform is None:
            raise ValueError("transform is required for vae_image (visual_gen) samples.")
        if vit_transform is None:
            # allow vit_transform missing: fall back to transform for vit_image
            vit_transform = transform

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.data_status = data_status

        self.data_dir_list = data_dir_list
        self.jsonl_path_list = jsonl_path_list

        self.transform = transform
        self.vit_transform = vit_transform
        # Deprecated compatibility knob. The default path now matches inference:
        # each first-seen image is inserted once, immediately before the ChatML
        # utterance for the turn where it first appears.
        self.only_current_turn_images = bool(only_current_turn_images)

        self.shuffle_lines = shuffle_lines
        self.shuffle_seed = shuffle_seed

        self._epoch_seed = 0

        # For compatibility with other datasets; not used for strict length limiting here.
        self.num_used_data = num_used_data

    def set_epoch(self, seed: int):
        self._epoch_seed = int(seed)

    def _get_worker_info(self) -> Tuple[int, int]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return 0, 1
        return info.id, info.num_workers

    def _shard_filter(self, global_index: int, worker_id: int, worker_count: int) -> bool:
        # Deterministic sharding across ranks and dataloader workers
        shard_count = self.world_size * worker_count
        shard_id = self.local_rank * worker_count + worker_id
        return (global_index % shard_count) == shard_id

    def _resolve_image_path(self, rel_path: str) -> str:
        # Use the first data_dir as root; support rel paths like "image/xxx.jpg"
        root = self.data_dir_list[0]
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(root, rel_path)

    def _load_pil(self, path: str) -> Image.Image:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(path)
        img = pil_img2rgb(img)
        return img

    def _encode_text(self, text: str) -> List[int]:
        # IMPORTANT: PackedDataset will add BOS and IM_END itself.
        return self.tokenizer.encode(text)

    def _make_text_plan(
        self,
        text: str,
        loss: int,
        enable_cfg: int,
        ce_loss_weight: float = 1.0,
        ce_loss_weights: Optional[List[float]] = None,
    ) -> Tuple[List[int], Dict]:
        text_ids = self._encode_text(text)
        plan = {
            "type": "text",
            "loss": int(loss),
            "enable_cfg": int(enable_cfg),
            "ce_loss_weight": float(ce_loss_weight),
            "special_token_loss": 0,
            "special_token_label": 0,
            "split_start": True,
            "split_end": True,
        }
        if ce_loss_weights is not None:
            plan["ce_loss_weights"] = [float(x) for x in ce_loss_weights]
        return text_ids, plan

    def _make_vit_image_plan(self, img_tensor: torch.Tensor, enable_cfg: int) -> Dict:
        return {
            "type": "vit_image",
            "loss": 0,
            "enable_cfg": int(enable_cfg),
            "special_token_loss": 0,
            "special_token_label": 0,
            "split_start": True,
            "split_end": True,
        }

    def _make_vae_image_plan(self, img_tensor: torch.Tensor, loss: int, enable_cfg: int) -> Dict:
        return {
            "type": "vae_image",
            "loss": int(loss),
            "enable_cfg": int(enable_cfg),
            "special_token_loss": 0,
            "special_token_label": 0,
            "split_start": True,
            "split_end": True,
        }

    def _estimate_tokens(self, sequence_plan: List[Dict], text_ids_list: List[List[int]], image_tensor_list: List[torch.Tensor]) -> int:
        # Upper-bound estimate used by PackedDataset length filters.
        # We count raw tokens + image token counts, excluding extra start/end tokens (PackedDataset adds +2*len(sequence_plan)).
        t_ptr = 0
        i_ptr = 0
        total = 0
        for item in sequence_plan:
            if item["type"] == "text":
                total += len(text_ids_list[t_ptr]) + 2  # BOS + IM_END approx
                t_ptr += 1
            elif item["type"] == "vit_image":
                img = image_tensor_list[i_ptr]
                # vit tokens = (H/patch)*(W/patch); patch_size is handled by patchify in PackedDataset
                # we approximate with max possible tokens by using raw pixels / 14.
                H, W = img.shape[1], img.shape[2]
                total += (H // 14) * (W // 14)
                i_ptr += 1
            elif item["type"] == "vae_image":
                img = image_tensor_list[i_ptr]
                # vae tokens = (H/downsample)*(W/downsample); downsample is configured in DataConfig (e.g., 16)
                # we approximate with raw pixels / 16 here.
                H, W = img.shape[1], img.shape[2]
                total += (H // 16) * (W // 16)
                i_ptr += 1
        return int(total)

    @staticmethod
    def _format_chatml_utterance(role: str, text: str) -> str:
        if role not in {"user", "assistant"}:
            role = "user"
        if not isinstance(text, str):
            text = ""
        return f"<|im_start|>{role}\n{text}\n<|im_end|>\n"

    @staticmethod
    def _format_chatml_assistant_prefix() -> str:
        return "<|im_start|>assistant\n"

    @staticmethod
    def _messages_to_plain_text(messages: List[dict]) -> str:
        """Keep message text exactly as text, including Image #N mentions."""
        lines = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content", "")
            if content is None:
                continue
            content = str(content)
            if content:
                lines.append(content)
        return "\n".join(lines).strip()


    @staticmethod
    def _valid_image_paths(images: List[str]) -> List[str]:
        return [p for p in images if isinstance(p, str) and p]

    def _resolve_image_messages(self, messages: List[dict], images: List[str]) -> Tuple[List[dict], List[str]]:
        """
        Attach concrete paths to image messages, but do not replace their text.
        Any unused images are still treated as turn-level image inputs.
        """
        if not isinstance(messages, list):
            messages = []
        if not isinstance(images, list):
            images = []

        resolved = []
        img_ptr = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            m_resolved = dict(m)
            if m_resolved.get("type", "") == "image":
                rel_path = None
                while img_ptr < len(images):
                    cand = images[img_ptr]
                    img_ptr += 1
                    if isinstance(cand, str) and cand:
                        rel_path = cand
                        break
                if isinstance(rel_path, str) and rel_path:
                    m_resolved["_bagel_image_path"] = rel_path
            resolved.append(m_resolved)

        return resolved, self._valid_image_paths(images[img_ptr:])

    def _image_dedupe_key(self, rel_path: str) -> str:
        try:
            return os.path.normcase(os.path.abspath(self._resolve_image_path(rel_path)))
        except Exception:
            return str(rel_path)

    def _turn_to_sample(
        self,
        episode: Dict,
        turns: List[Dict],
        turn_idx: int,
        global_index: int,
        worker_id: int,
    ) -> Optional[Dict]:
        if turn_idx < 0 or turn_idx >= len(turns):
            return None
        turn = turns[turn_idx]
        task_type = turn.get("task_type", None)
        render = turn.get("render", None)
        if task_type is None or render is None or not isinstance(render, dict):
            return None
        if task_type not in {"vlm_understand", "select_mcq", "plan_struct", "t2i", "edit", "interleaved_mm"}:
            return None

        image_tensor_list: List[torch.Tensor] = []
        text_ids_list: List[List[int]] = []
        sequence_plan: List[Dict] = []
        seen_input_images: Dict[str, int] = {}

        def add_text_block(
            text: str,
            loss: int,
            enable_cfg: int = 1,
            ce_loss_weight: float = 1.0,
            ce_loss_weights: Optional[List[float]] = None,
        ):
            ids, plan = self._make_text_plan(
                text,
                loss=loss,
                enable_cfg=enable_cfg,
                ce_loss_weight=ce_loss_weight,
                ce_loss_weights=ce_loss_weights,
            )
            text_ids_list.append(ids)
            sequence_plan.append(plan)

        def add_vit_image(rel_path: str, enable_cfg: int = 1):
            img_path = self._resolve_image_path(rel_path)
            img = self._load_pil(img_path)
            img_tensor = self.vit_transform(img)
            image_tensor_list.append(img_tensor)
            sequence_plan.append(self._make_vit_image_plan(img_tensor, enable_cfg=enable_cfg))

        def add_vae_image(rel_path: str, loss: int, enable_cfg: int = 1):
            img_path = self._resolve_image_path(rel_path)
            img = self._load_pil(img_path)
            img_tensor = self.transform(img)
            image_tensor_list.append(img_tensor)
            sequence_plan.append(self._make_vae_image_plan(img_tensor, loss=loss, enable_cfg=enable_cfg))

        current_ra = (render.get("answer", {}) or {}) if isinstance(render.get("answer", None), dict) else {}
        current_ra_images = current_ra.get("images", []) or []
        current_ra_messages = current_ra.get("messages", []) or []
        current_has_answer_image = any(
            isinstance(m, dict) and m.get("type", "") == "image" for m in current_ra_messages
        ) or len(self._valid_image_paths(current_ra_images)) > 0
        generation_round = task_type in {"t2i", "edit"} or (task_type == "interleaved_mm" and current_has_answer_image)

        def add_context_image(rel_path: str):
            # Matches InterleaveInferencer.update_context_image():
            # generation rounds condition on VAE+ViT, understanding rounds on ViT.
            if generation_round:
                add_vae_image(rel_path, loss=0, enable_cfg=1)
            add_vit_image(rel_path, enable_cfg=1)

        def add_latent_context(rel_path: str):
            # LatentCOT hidden history: previous outputs are encoded through
            # the VAE path only and are not exposed as visible image/text chat.
            add_vae_image(rel_path, loss=0, enable_cfg=1)

        def add_images_before_utterance(messages: List[dict], extra_images: List[str]) -> List[int]:
            image_paths: List[str] = []
            for m in messages:
                if not isinstance(m, dict) or m.get("type", "") != "image":
                    continue
                rel_path = m.get("_bagel_image_path", None)
                if isinstance(rel_path, str) and rel_path:
                    image_paths.append(rel_path)
            image_paths.extend(self._valid_image_paths(extra_images))

            placeholder_numbers: List[int] = []
            for rel_path in image_paths:
                key = self._image_dedupe_key(rel_path)
                if key in seen_input_images:
                    continue
                image_no = len(seen_input_images) + 1
                seen_input_images[key] = image_no
                add_context_image(rel_path)
                placeholder_numbers.append(image_no)
            return placeholder_numbers

        def add_chatml_utterance(role: str, messages: List[dict], images: List[str]):
            resolved_messages, remaining_images = self._resolve_image_messages(messages, images)
            add_images_before_utterance(resolved_messages, remaining_images)
            text = self._messages_to_plain_text(resolved_messages)
            if text:
                add_text_block(self._format_chatml_utterance(role, text), loss=0, enable_cfg=1)

        latent_context = turn.get("latent_context", []) or []
        if isinstance(latent_context, list) and len(latent_context) > 0:
            rq = render.get("query", {}) or {}
            if not isinstance(rq, dict):
                rq = {}
            add_chatml_utterance("user", rq.get("messages", []) or [], rq.get("images", []) or [])
            for ctx in latent_context:
                if not isinstance(ctx, dict):
                    continue
                rel_path = ctx.get("latent", None) or ctx.get("image", None)
                if isinstance(rel_path, str) and rel_path:
                    add_latent_context(rel_path)
        else:
            # Build full history plus current user query in the same order a caller
            # would feed InterleaveInferencer: images first, then the ChatML text
            # for the turn where those images first appear.
            for hidx in range(turn_idx + 1):
                hturn = turns[hidx]
                if not isinstance(hturn, dict):
                    continue
                hrender = hturn.get("render", None)
                if not isinstance(hrender, dict):
                    continue

                hrq = hrender.get("query", {}) or {}
                hra = hrender.get("answer", {}) or {}
                if not isinstance(hrq, dict):
                    hrq = {}
                if not isinstance(hra, dict):
                    hra = {}

                add_chatml_utterance("user", hrq.get("messages", []) or [], hrq.get("images", []) or [])

                if hidx == turn_idx:
                    continue

                add_chatml_utterance("assistant", hra.get("messages", []) or [], hra.get("images", []) or [])

        add_text_block(self._format_chatml_assistant_prefix(), loss=0, enable_cfg=1)

        # -----------------
        # Build ANSWER (with loss)
        # -----------------
        if task_type in {"vlm_understand", "select_mcq", "plan_struct"}:
            ra = render.get("answer", {}) or {}
            if not isinstance(ra, dict):
                ra = {}
            answer_text = self._messages_to_plain_text(ra.get("messages", []) or [])
            if not answer_text:
                return None
            add_text_block(answer_text, loss=1, enable_cfg=0)

        elif task_type == "t2i":
            ra = render.get("answer", {}) or {}
            if not isinstance(ra, dict):
                ra = {}
            ra_images = ra.get("images", []) or []
            if len(ra_images) == 0 or not isinstance(ra_images[0], str) or not ra_images[0]:
                return None
            add_vae_image(ra_images[0], loss=1, enable_cfg=0)

        elif task_type == "edit":
            ra = render.get("answer", {}) or {}
            if not isinstance(ra, dict):
                ra = {}
            ra_images = ra.get("images", []) or []
            if len(ra_images) == 0 or not isinstance(ra_images[0], str) or not ra_images[0]:
                return None
            add_vae_image(ra_images[0], loss=1, enable_cfg=0)

        elif task_type == "interleaved_mm":
            ra = render.get("answer", {}) or {}
            if not isinstance(ra, dict):
                ra = {}
            ra_images = ra.get("images", []) or []
            ra_messages, remaining_answer_images = self._resolve_image_messages(
                ra.get("messages", []) or [],
                ra_images,
            )
            img_ptr = 0
            target_items = 0
            for m in ra_messages:
                if not isinstance(m, dict):
                    continue
                mtype = m.get("type", "")
                if mtype == "text":
                    content = m.get("content", "")
                    if content:
                        if content in {"<latent_continue>", "<latent_final>"}:
                            ids = self._encode_text(content)
                            control_id = self.tokenizer.convert_tokens_to_ids(content)
                            ce_weights = [100.0 if label_id == control_id else 0.0 for label_id in ids + [self.tokenizer.eos_token_id]]
                            add_text_block(content, loss=1, enable_cfg=0, ce_loss_weights=ce_weights)
                        else:
                            add_text_block(content, loss=1, enable_cfg=0)
                        target_items += 1
                elif mtype == "image":
                    rel_path = m.get("_bagel_image_path", None)
                    if not (isinstance(rel_path, str) and rel_path):
                        if img_ptr >= len(remaining_answer_images):
                            return None
                        rel_path = remaining_answer_images[img_ptr]
                        img_ptr += 1
                    add_vae_image(rel_path, loss=1, enable_cfg=0)
                    target_items += 1

            if target_items == 0:
                return None

        else:
            return None

        num_tokens = self._estimate_tokens(sequence_plan, text_ids_list, image_tensor_list)
        sample = {
            "image_tensor_list": image_tensor_list,
            "text_ids_list": text_ids_list,
            "sequence_plan": sequence_plan,
            "num_tokens": num_tokens,
            "data_indexes": {
                "dataset_name": self.dataset_name,
                "worker_id": int(worker_id),
                "data_indexes": [int(global_index), str(episode.get("id", "")), int(turn_idx + 1), str(task_type)],
            },
        }
        return sample

    def __iter__(self) -> Iterator[Dict]:
        worker_id, worker_count = self._get_worker_info()

        # Collect all lines (for strict determinism and easy sharding). If file is huge, you can stream instead.
        # Here we keep it simple; minimal changes later if needed.
        episodes: List[Dict] = []
        for p in self.jsonl_path_list:
            if not os.path.exists(p):
                raise FileNotFoundError(f"jsonl not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    episodes.append(json.loads(line))

        indices = list(range(len(episodes)))
        if self.shuffle_lines:
            rnd = random.Random(self.shuffle_seed + self._epoch_seed)
            rnd.shuffle(indices)

        global_index = 0
        while True:
            for idx in indices:
                if not self._shard_filter(global_index, worker_id, worker_count):
                    global_index += 1
                    continue

                episode = episodes[idx]
                turns = episode.get("turns", [])
                if not isinstance(turns, list) or len(turns) == 0:
                    global_index += 1
                    continue

                for turn_idx, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        continue
                    sample = self._turn_to_sample(
                        episode,
                        turns,
                        turn_idx=turn_idx,
                        global_index=global_index,
                        worker_id=worker_id,
                    )
                    if sample is not None:
                        yield sample

                global_index += 1
