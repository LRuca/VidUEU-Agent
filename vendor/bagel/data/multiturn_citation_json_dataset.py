# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from .citation_dialogue_utils import (
    CE_GROUP_ANSWER_TEXT,
    CE_GROUP_CITE,
    CE_GROUP_CLAIM,
    GEN_CITATION_SYSTEM_PROMPT,
    MSE_GROUP_ANSWER_IMAGE,
    VLM_CITATION_SYSTEM_PROMPT,
    build_cite_block,
    build_claim_block,
    build_evidence_index,
    classify_assistant_round,
    collect_first_appearance_images_for_text,
    extract_turn_image_maps,
    format_chatml_assistant_prefix,
    format_chatml_utterance,
    get_first_valid_image_uri,
    merge_image_uri_map,
    remove_citation_markup_from_history_text,
    resolve_turn_id,
)
from .data_utils import pil_img2rgb


class _SkipTurnError(RuntimeError):
    pass


class _SkipDialogueError(RuntimeError):
    pass


class MultiTurnCitationJSONIterableDataset(torch.utils.data.IterableDataset):
    """
    Citation-mode multi-turn dataset for WOOL/WEAVE-style dialogue JSON.

    Supported file format:
    - JSON array: [ {...}, {...} ]
    - JSONL: one dialogue object per line

    Each dialogue object must contain:
      {
        "turns": [
          {
            "turn_id": ...,
            "user": {...},
            "assistant": {...}
          },
          ...
        ]
      }
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
        shuffle_lines: bool = True,
        shuffle_seed: int = 0,
        num_used_data: Optional[List[int]] = None,
        strict_evidence: bool = True,
        max_warnings: int = 20,
        include_turn_in_context: bool = False,
        **kwargs,
    ):
        super().__init__()
        if jsonl_path_list is None or len(jsonl_path_list) == 0:
            raise ValueError("jsonl_path_list is required for MultiTurnCitationJSONIterableDataset.")
        if transform is None:
            raise ValueError("transform is required for vae_image (visual_gen) samples.")
        if vit_transform is None:
            vit_transform = transform

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.data_status = data_status

        self.data_dir_list = data_dir_list or []
        self.jsonl_path_list = jsonl_path_list

        self.transform = transform
        self.vit_transform = vit_transform
        self.shuffle_lines = bool(shuffle_lines)
        self.shuffle_seed = int(shuffle_seed)
        self._epoch_seed = 0

        self.num_used_data = num_used_data
        self.strict_evidence = bool(strict_evidence)
        self.max_warnings = int(max_warnings)
        self._warn_n = 0
        self.include_turn_in_context = bool(include_turn_in_context)

    def _warn(self, message: str):
        self._warn_n += 1
        if self.local_rank == 0 and self._warn_n <= self.max_warnings:
            print(f"[{self.dataset_name}] {message}", flush=True)

    def set_epoch(self, seed: int):
        self._epoch_seed = int(seed)

    def _get_worker_info(self) -> Tuple[int, int]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return 0, 1
        return info.id, info.num_workers

    def _shard_filter(self, global_index: int, worker_id: int, worker_count: int) -> bool:
        shard_count = self.world_size * worker_count
        shard_id = self.local_rank * worker_count + worker_id
        return (global_index % shard_count) == shard_id

    def _resolve_image_path(self, path: str) -> str:
        if not isinstance(path, str) or not path:
            return ""
        if os.path.isabs(path):
            return path
        if len(self.data_dir_list) > 0 and isinstance(self.data_dir_list[0], str) and self.data_dir_list[0]:
            return os.path.join(self.data_dir_list[0], path)
        return path

    def _load_pil(self, path: str) -> Image.Image:
        path = self._resolve_image_path(path)
        if not path or (not os.path.exists(path)):
            raise _SkipDialogueError(f"image_not_found: {path}")
        try:
            img = Image.open(path)
            img = pil_img2rgb(img)
            return img
        except Exception as e:
            raise _SkipDialogueError(f"image_open_failed: {path} ({e})") from e

    def _encode_text(self, text: str) -> List[int]:
        if not isinstance(text, str):
            text = ""
        # Use raw text tokenization only. PackedDataset adds BOS/EOS itself.
        try:
            enc = self.tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = enc.get("input_ids", None)
            if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
                return input_ids[0]
            if isinstance(input_ids, list):
                return input_ids
        except TypeError:
            pass
        except Exception:
            pass

        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return self.tokenizer.encode(text)

    def _make_text_plan_from_ids(
        self,
        text_ids: List[int],
        loss: int,
        enable_cfg: int,
        ce_group_ids: Optional[List[int]] = None,
    ) -> Dict:
        plan = {
            "type": "text",
            "loss": int(loss),
            "enable_cfg": int(enable_cfg),
            "special_token_loss": 0,
            "special_token_label": 0,
            "split_start": True,
            "split_end": True,
        }
        if ce_group_ids is not None:
            if len(ce_group_ids) != (len(text_ids) + 1):
                raise _SkipTurnError(
                    f"ce_group_ids_length_mismatch: expected={len(text_ids)+1}, got={len(ce_group_ids)}"
                )
            plan["ce_group_ids"] = [int(x) for x in ce_group_ids]
        return plan

    def _make_vit_image_plan(self, enable_cfg: int) -> Dict:
        return {
            "type": "vit_image",
            "loss": 0,
            "enable_cfg": int(enable_cfg),
            "special_token_loss": 0,
            "special_token_label": 0,
            "split_start": True,
            "split_end": True,
        }

    def _make_vae_image_plan(self, loss: int, enable_cfg: int, mse_group_id: Optional[int] = None) -> Dict:
        plan = {
            "type": "vae_image",
            "loss": int(loss),
            "enable_cfg": int(enable_cfg),
            "special_token_loss": 0,
            "special_token_label": 0,
            "split_start": True,
            "split_end": True,
        }
        if mse_group_id is not None:
            plan["mse_group_id"] = int(mse_group_id)
        return plan

    def _estimate_tokens(
        self,
        sequence_plan: List[Dict],
        text_ids_list: List[List[int]],
        image_tensor_list: List[torch.Tensor],
    ) -> int:
        t_ptr = 0
        i_ptr = 0
        total = 0
        for item in sequence_plan:
            if item["type"] == "text":
                total += len(text_ids_list[t_ptr]) + 2
                t_ptr += 1
            elif item["type"] == "vit_image":
                img = image_tensor_list[i_ptr]
                h, w = img.shape[1], img.shape[2]
                total += (h // 14) * (w // 14)
                i_ptr += 1
            elif item["type"] == "vae_image":
                img = image_tensor_list[i_ptr]
                h, w = img.shape[1], img.shape[2]
                total += (h // 16) * (w // 16)
                i_ptr += 1
        return int(total)

    def _build_grouped_target_text(
        self,
        cite_block: str,
        claim_block: str,
        answer_text: Optional[str],
    ) -> Tuple[str, List[int], List[int]]:
        segments: List[Tuple[int, str]] = [
            (CE_GROUP_CITE, cite_block),
            (CE_GROUP_CLAIM, claim_block),
        ]
        if isinstance(answer_text, str) and len(answer_text) > 0:
            segments.append((CE_GROUP_ANSWER_TEXT, answer_text))

        # Keep exact target formatting: <cite>...</cite><claim>...</claim>answer
        # Group attribution is defined over segment-wise tokenization so we can
        # supervise cite/claim/answer independently even when byte-BPE merges
        # across boundaries in full-string tokenization.
        full_text = "".join([text for _, text in segments])
        part_ids = [self._encode_text(text) for _, text in segments]

        flat_ids: List[int] = []
        token_groups: List[int] = []
        for (gid, _), ids in zip(segments, part_ids):
            flat_ids.extend(ids)
            token_groups.extend([gid] * len(ids))

        eos_group = segments[-1][0]
        label_groups = token_groups + [eos_group]
        return full_text, flat_ids, label_groups

    def _iter_file_dialogues(self, path: str) -> List[dict]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"json/jsonl not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            first_non_ws = ""
            while True:
                ch = f.read(1)
                if ch == "":
                    break
                if not ch.isspace():
                    first_non_ws = ch
                    break
            f.seek(0)

            if first_non_ws == "[":
                obj = json.load(f)
                if isinstance(obj, list):
                    return [x for x in obj if isinstance(x, dict)]
                if isinstance(obj, dict):
                    return [obj]
                return []

            out: List[dict] = []
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    self._warn(f"jsonl_decode_failed path={path} line={line_no}")
                    continue
                if isinstance(item, dict):
                    out.append(item)
            return out

    def _build_sample_for_turn(
        self,
        episode: dict,
        turns: Sequence[dict],
        turn_idx: int,
        global_index: int,
        worker_id: int,
    ) -> Optional[Dict]:
        curr_turn = turns[turn_idx]
        if not isinstance(curr_turn, dict):
            return None

        curr_turn_id = resolve_turn_id(curr_turn, turn_idx)
        curr_user = curr_turn.get("user", {}) or {}
        curr_assistant = curr_turn.get("assistant", {}) or {}
        round_type = classify_assistant_round(curr_assistant)
        if round_type not in {"text", "image"}:
            raise _SkipTurnError(f"unsupported_round_type turn={curr_turn_id}")

        image_tensor_list: List[torch.Tensor] = []
        text_ids_list: List[List[int]] = []
        sequence_plan: List[Dict] = []

        def add_text_ids_block(text_ids: List[int], loss: int, enable_cfg: int, ce_group_ids: Optional[List[int]] = None):
            text_ids_list.append(text_ids)
            sequence_plan.append(
                self._make_text_plan_from_ids(text_ids=text_ids, loss=loss, enable_cfg=enable_cfg, ce_group_ids=ce_group_ids)
            )

        def add_text_block(text: str, loss: int, enable_cfg: int):
            text_ids = self._encode_text(text)
            add_text_ids_block(text_ids=text_ids, loss=loss, enable_cfg=enable_cfg, ce_group_ids=None)

        def add_vit_image(path: str, enable_cfg: int):
            img = self._load_pil(path)
            img_tensor = self.vit_transform(img)
            image_tensor_list.append(img_tensor)
            sequence_plan.append(self._make_vit_image_plan(enable_cfg=enable_cfg))

        def add_vae_image(path: str, loss: int, enable_cfg: int, mse_group_id: Optional[int] = None):
            img = self._load_pil(path)
            img_tensor = self.transform(img)
            image_tensor_list.append(img_tensor)
            sequence_plan.append(self._make_vae_image_plan(loss=loss, enable_cfg=enable_cfg, mse_group_id=mse_group_id))

        def add_context_image(path: str, enable_cfg: int):
            """
            Align with inferencer.update_context_image():
            - text round (understanding_output=True): ViT only
            - image round (understanding_output=False): VAE then ViT
            """
            img = self._load_pil(path)
            if round_type == "image":
                vae_img_tensor = self.transform(img)
                image_tensor_list.append(vae_img_tensor)
                sequence_plan.append(self._make_vae_image_plan(loss=0, enable_cfg=enable_cfg, mse_group_id=None))

            vit_img_tensor = self.vit_transform(img)
            image_tensor_list.append(vit_img_tensor)
            sequence_plan.append(self._make_vit_image_plan(enable_cfg=enable_cfg))

        # Query system prompt (no loss).
        system_prompt = VLM_CITATION_SYSTEM_PROMPT if round_type == "text" else GEN_CITATION_SYSTEM_PROMPT
        add_text_block(system_prompt, loss=0, enable_cfg=1)

        # Build multi-turn query with first-appearance image insertion.
        inserted_image_nums = set()
        known_image_uri_map: Dict[int, str] = {}

        def add_query_utterance(turn_no: int, role: str, text: str):
            if not isinstance(text, str):
                text = ""
            first_images = collect_first_appearance_images_for_text(
                text=text,
                image_uri_map=known_image_uri_map,
                inserted_image_nums=inserted_image_nums,
            )
            for it in first_images:
                uri = it.get("uri", "")
                if isinstance(uri, str) and uri:
                    add_context_image(uri, enable_cfg=1)

            utterance = format_chatml_utterance(
                role=role,
                text=text,
                include_turn=self.include_turn_in_context,
                turn_no=turn_no,
            )
            add_text_block(utterance, loss=0, enable_cfg=1)

        for hidx in range(turn_idx + 1):
            hturn = turns[hidx]
            if not isinstance(hturn, dict):
                continue
            hturn_id = resolve_turn_id(hturn, hidx)
            huser = hturn.get("user", {}) or {}
            hassistant = hturn.get("assistant", {}) or {}

            h_user_map, h_assistant_map = extract_turn_image_maps(hturn)
            merge_image_uri_map(known_image_uri_map, h_user_map)
            add_query_utterance(hturn_id, "user", huser.get("text", "") or "")

            if hidx == turn_idx:
                continue

            merge_image_uri_map(known_image_uri_map, h_assistant_map)
            hist_assistant_text = remove_citation_markup_from_history_text(hassistant.get("text", "") or "")
            if (not hist_assistant_text) and len(h_assistant_map) > 0:
                first_num = list(h_assistant_map.keys())[0]
                hist_assistant_text = f"Image #{first_num}"
            add_query_utterance(hturn_id, "assistant", hist_assistant_text)

        add_text_block(format_chatml_assistant_prefix(), loss=0, enable_cfg=1)

        citations = curr_assistant.get("citations", {}) or {}
        citation_eids = citations.get("evidence_ids", []) or []
        if not isinstance(citation_eids, list):
            citation_eids = []
        citation_eids = [x for x in citation_eids if isinstance(x, int)]
        if len(citation_eids) == 0:
            raise _SkipTurnError(f"empty_citation_eids turn={curr_turn_id}")

        claims = curr_assistant.get("claims", []) or []
        if not isinstance(claims, list) or len(claims) == 0:
            raise _SkipTurnError(f"empty_claims turn={curr_turn_id}")

        evidence_index, evidence_errors = build_evidence_index(turns, turn_idx)
        if self.strict_evidence and len(evidence_errors) > 0:
            raise _SkipTurnError(f"evidence_index_errors turn={curr_turn_id} errors={evidence_errors[:3]}")

        cite_block, citation_local_idx_map, cite_errors = build_cite_block(citation_eids, evidence_index)
        if len(cite_errors) > 0:
            raise _SkipTurnError(f"cite_build_errors turn={curr_turn_id} errors={cite_errors[:3]}")

        claim_block, claim_errors = build_claim_block(
            claims=claims,
            citation_local_idx_map=citation_local_idx_map,
            strict_eids=True,
        )
        if len(claim_errors) > 0:
            raise _SkipTurnError(f"claim_build_errors turn={curr_turn_id} errors={claim_errors[:3]}")

        if round_type == "text":
            answer_text = curr_assistant.get("text", "") or ""
            if not isinstance(answer_text, str) or len(answer_text.strip()) == 0:
                raise _SkipTurnError(f"empty_text_answer turn={curr_turn_id}")
            full_target_text, full_target_ids, label_group_ids = self._build_grouped_target_text(
                cite_block=cite_block,
                claim_block=claim_block,
                answer_text=answer_text,
            )
            _ = full_target_text  # explicit for readability
            add_text_ids_block(text_ids=full_target_ids, loss=1, enable_cfg=0, ce_group_ids=label_group_ids)

        else:
            full_target_text, full_target_ids, label_group_ids = self._build_grouped_target_text(
                cite_block=cite_block,
                claim_block=claim_block,
                answer_text=None,
            )
            _ = full_target_text
            add_text_ids_block(text_ids=full_target_ids, loss=1, enable_cfg=0, ce_group_ids=label_group_ids)

            answer_image_uri = get_first_valid_image_uri(curr_assistant.get("images_out", []) or [])
            if not (isinstance(answer_image_uri, str) and answer_image_uri):
                raise _SkipTurnError(f"missing_answer_image turn={curr_turn_id}")
            add_vae_image(
                answer_image_uri,
                loss=1,
                enable_cfg=0,
                mse_group_id=MSE_GROUP_ANSWER_IMAGE,
            )

        num_tokens = self._estimate_tokens(sequence_plan, text_ids_list, image_tensor_list)
        sample = {
            "image_tensor_list": image_tensor_list,
            "text_ids_list": text_ids_list,
            "sequence_plan": sequence_plan,
            "num_tokens": num_tokens,
            "data_indexes": {
                "dataset_name": self.dataset_name,
                "worker_id": int(worker_id),
                "data_indexes": [
                    int(global_index),
                    str(episode.get("id", "")),
                    int(curr_turn_id),
                    str(round_type),
                ],
            },
        }
        return sample

    def __iter__(self) -> Iterator[Dict]:
        worker_id, worker_count = self._get_worker_info()

        episodes: List[dict] = []
        for path in self.jsonl_path_list:
            episodes.extend(self._iter_file_dialogues(path))
        if len(episodes) == 0:
            raise RuntimeError("No dialogue episodes loaded for MultiTurnCitationJSONIterableDataset.")

        indices = list(range(len(episodes)))
        if self.shuffle_lines:
            rnd = random.Random(self.shuffle_seed + self._epoch_seed)
            rnd.shuffle(indices)

        global_index = 0
        while True:
            yielded_this_round = 0
            for ep_idx in indices:
                if not self._shard_filter(global_index, worker_id, worker_count):
                    global_index += 1
                    continue

                episode = episodes[ep_idx]
                turns = episode.get("turns", []) if isinstance(episode, dict) else []
                if not isinstance(turns, list) or len(turns) == 0:
                    global_index += 1
                    continue

                skip_dialogue = False
                for turn_idx in range(len(turns)):
                    if skip_dialogue:
                        break
                    try:
                        sample = self._build_sample_for_turn(
                            episode=episode,
                            turns=turns,
                            turn_idx=turn_idx,
                            global_index=global_index,
                            worker_id=worker_id,
                        )
                    except _SkipDialogueError as e:
                        self._warn(f"skip_dialogue idx={ep_idx} reason={e}")
                        skip_dialogue = True
                        continue
                    except _SkipTurnError as e:
                        self._warn(f"skip_turn idx={ep_idx} turn={turn_idx+1} reason={e}")
                        continue
                    except Exception as e:
                        self._warn(f"skip_turn idx={ep_idx} turn={turn_idx+1} unexpected_error={e}")
                        continue

                    if sample is not None:
                        yielded_this_round += 1
                        yield sample

                global_index += 1
            if yielded_this_round == 0:
                raise RuntimeError(
                    "MultiTurnCitationJSONIterableDataset produced zero valid samples in one full pass. "
                    "Please check image paths and evidence fields."
                )
