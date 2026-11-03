# executor/tasks/image_embedding_diversity.py
"""
基于图像语义向量（embedding）的多样性工具。

功能：
- 使用 CLIP 模型为每一帧图片提取语义向量；
- 计算帧间余弦相似度；
- 给出“语义变化分数”（semantic_change），可作为关键帧选择的一路信号；
- 如有需要，也可以返回完整的相似度矩阵，用于可视化或更高级策略。

新增（最小修改）：
- 支持 compute_text_embeddings(texts)
  直接复用同一个 CLIP 模型的 text tower，避免重复加载新模型。
"""

from typing import Dict, Any, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ImageEmbeddingDiversity:
    """
    图像语义向量 + 多样性分析。
    """

    def __init__(self, config: Dict[str, Any]):
        models_conf = config.get("models", {})
        task_conf = config.get("tasks", {}).get("T2_image_embedding", {})

        # 👉 使用本地模型目录（例如 /home/rtjjjj/video_agent/models/openclip-b32）
        self.model_name = (
            task_conf.get("model_name")
            or models_conf.get("image_embedding_model")
        )

        # 是否强制只用本地文件
        self.local_files_only = bool(task_conf.get("local_files_only", True))

        # 设备：任务配置 > 全局配置 > 自动检测
        device = task_conf.get("device") or config.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.batch_size = int(task_conf.get("batch_size", 16))

        # [NEW] text embedding 相关配置
        self.text_batch_size = int(task_conf.get("text_batch_size", self.batch_size))
        self.max_text_tokens = int(task_conf.get("max_text_tokens", 77))

        print(
            f"[ImageEmbeddingDiversity] Loading model from local path: "
            f"{self.model_name} on {self.device} (local_files_only={self.local_files_only})"
        )

        try:
            # ⭐ 从本地目录加载
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        except Exception as e:
            print(f"⚠️ [ImageEmbeddingDiversity] Failed to load CLIP model locally: {e}")
            if "torch.load" in str(e) and "at least v2.6" in str(e):
                print(
                    "⚠️ [ImageEmbeddingDiversity] Current local CLIP weights appear to require "
                    "`torch.load`, which transformers now blocks on torch<2.6. "
                    "Use safetensors weights, upgrade torch to >=2.6, or let the pipeline skip "
                    "image embedding for now."
                )
            raise

        self.model.to(self.device)
        self.model.eval()
        print("[ImageEmbeddingDiversity] Model loaded.")

    # ==================================================================
    # 基础工具函数
    # ==================================================================

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """安全地加载并转换为 RGB。"""
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """对矩阵的每一行做 L2 归一化。"""
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.maximum(norm, eps)
        return x / norm

    @staticmethod
    def _cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
        """
        emb: shape = (N, D) 的 L2 归一化向量。
        返回 N×N 的余弦相似度矩阵。
        """
        if emb.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return emb @ emb.T

    @staticmethod
    def _clean_text(text: str) -> str:
        """最小文本清洗：去首尾空白、合并多余空格。"""
        s = str(text or "").strip()
        if not s:
            return ""
        s = " ".join(s.split())
        return s

    def _projection_dim(self) -> int:
        """推断 CLIP 投影维度。"""
        try:
            return int(getattr(self.model.config, "projection_dim"))
        except Exception:
            pass

        try:
            return int(self.model.text_projection.out_features)
        except Exception:
            pass

        return 512

    # ==================================================================
    # 主接口：计算 image embeddings & 语义变化
    # ==================================================================

    @torch.no_grad()
    def compute_embeddings(
        self,
        frame_paths: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        对一串帧路径计算图像语义向量（未做 L2 归一化）。

        返回值：
            embeddings: np.ndarray, shape = (N, D)
        """
        if batch_size is None:
            batch_size = self.batch_size

        n = len(frame_paths)
        if n == 0:
            return np.zeros((0, 0), dtype=np.float32)

        all_embeds: List[np.ndarray] = []

        for i in range(0, n, batch_size):
            batch_paths = frame_paths[i : i + batch_size]
            images: List[Image.Image] = []
            for p in batch_paths:
                try:
                    images.append(self._load_image(p))
                except Exception as e:
                    print(f"⚠️ [ImageEmbeddingDiversity] Failed to load image: {p} ({e})")
                    images.append(Image.new("RGB", (224, 224), (0, 0, 0)))

            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            image_features = self.model.get_image_features(**inputs)
            if hasattr(image_features, "image_embeds"):
                image_features = image_features.image_embeds
            elif hasattr(image_features, "pooler_output"):
                image_features = image_features.pooler_output
            elif hasattr(image_features, "last_hidden_state"):
                image_features = image_features.last_hidden_state[:, 0, :]

            image_features = image_features.detach().cpu().numpy().astype("float32")
            all_embeds.append(image_features)

            for im in images:
                try:
                    im.close()
                except Exception:
                    pass

        embeddings = np.concatenate(all_embeds, axis=0)
        return embeddings

    @torch.no_grad()
    def compute_text_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        对一组文本计算 CLIP text embeddings（未做 L2 归一化）。

        返回：
            np.ndarray, shape = (N, D)
        """
        if batch_size is None:
            batch_size = self.text_batch_size

        n = len(texts)
        d = self._projection_dim()

        if n == 0:
            return np.zeros((0, d), dtype=np.float32)

        cleaned_texts: List[str] = []
        empty_mask: List[bool] = []

        for t in texts:
            s = self._clean_text(t)
            is_empty = (len(s) == 0)
            empty_mask.append(is_empty)

            # CLIP 不能直接处理完全空输入，先用占位符跑 tokenizer，
            # 后面再把该位置 embedding 清零。
            cleaned_texts.append(s if not is_empty else ".")

        all_embeds: List[np.ndarray] = []

        for i in range(0, n, batch_size):
            batch_texts = cleaned_texts[i : i + batch_size]

            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_text_tokens,
            ).to(self.device)

            text_features = self.model.get_text_features(**inputs)
            if hasattr(text_features, "text_embeds"):
                text_features = text_features.text_embeds
            elif hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            elif hasattr(text_features, "last_hidden_state"):
                text_features = text_features.last_hidden_state[:, 0, :]

            text_features = text_features.detach().cpu().numpy().astype("float32")
            all_embeds.append(text_features)

        embeddings = np.concatenate(all_embeds, axis=0)

        # 原始空文本位置强制置零，避免 "." 产生伪语义
        for i, is_empty in enumerate(empty_mask):
            if is_empty:
                embeddings[i, :] = 0.0

        return embeddings

    def compute_semantic_change(
        self,
        embeddings: np.ndarray,
    ) -> List[float]:
        """
        根据相邻帧的 cosine 相似度，计算“语义变化分数”。

        定义：
            sim_i = cos(emb_{i-1}, emb_i)
            change_i = 1 - sim_i     (i >= 1)
            change_0 = 0.0           (第一帧没有前驱)

        返回：
            List[float]，长度 = N
        """
        if embeddings.ndim != 2:
            raise ValueError(f"Expected shape (N, D), got {embeddings.shape}")

        n = embeddings.shape[0]
        if n == 0:
            return []

        emb_norm = self._l2_normalize(embeddings)

        changes: List[float] = [0.0] * n
        for i in range(1, n):
            v_prev = emb_norm[i - 1]
            v_cur = emb_norm[i]
            sim = float(np.clip(v_prev @ v_cur, -1.0, 1.0))
            changes[i] = 1.0 - sim

        return changes

    @torch.no_grad()
    def compute_semantic_change_for_frame_paths(
        self,
        frame_paths: List[str],
        batch_size: Optional[int] = None,
        return_similarity_matrix: bool = False,
    ) -> Dict[str, Any]:
        """
        综合接口：给帧路径列表，直接算出：
          - embeddings
          - semantic_change（相邻帧语义变化）
          - 可选：full_similarity（完整 N×N 相似度矩阵）

        返回：
            {
              "embeddings": np.ndarray,        # shape = (N, D)
              "change_scores": List[float],    # len = N
              "similarity": np.ndarray | None, # shape = (N, N) if requested
            }
        """
        embeddings = self.compute_embeddings(frame_paths, batch_size=batch_size)
        change_scores = self.compute_semantic_change(embeddings)

        sim_mat = None
        if return_similarity_matrix:
            emb_norm = self._l2_normalize(embeddings)
            sim_mat = self._cosine_similarity_matrix(emb_norm)

        return {
            "embeddings": embeddings,
            "change_scores": change_scores,
            "similarity": sim_mat,
        }