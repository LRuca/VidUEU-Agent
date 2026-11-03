# executor/tasks/optical_flow_raft.py

import os
import sys
from typing import List, Optional

import cv2
import numpy as np
import torch


class RAFTArgs(dict):
    """
    RAFT 参数包装：

    - 既是 dict，可以用 'key' in args、for k in args
    - 又支持属性访问：args.small 等价于 args['small']
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self


class RAFTOpticalFlow:
    """
    使用本地 RAFT 仓库做光流推理的封装（修正版）

    - 在 CPU、单卡和多卡环境中都直接加载基础 RAFT 模型
    - 自动兼容带 ``module.`` 前缀的 DataParallel checkpoint
    - 每一对帧失败时只报 WARN，并返回 0.0，不中断整个 pipeline
    """

    def __init__(
        self,
        repo_root: str,
        weights: str = "models/raft-things.pth",
        device: str = "cuda",
        small: bool = False,
        mixed_precision: bool = True,
        alternate_corr: bool = False,
    ):
        """
        :param repo_root: RAFT 仓库根目录，例如 /home/rtjjjj/guru_video/third_party/raft
        :param weights:   权重路径，相对路径会拼到 repo_root 下
        :param device:    "cuda" / "cuda:0" / "cpu"
        """
        self.repo_root = os.path.abspath(repo_root)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        core_dir = os.path.join(self.repo_root, "core")
        if not os.path.isdir(core_dir):
            raise FileNotFoundError(
                f"[RAFTOpticalFlow] core/ directory not found under repo_root: {core_dir}"
            )

        # 关键：把 repo_root 和 core/ 加到 sys.path，才能 import core.raft / utils.utils
        if self.repo_root not in sys.path:
            sys.path.insert(0, self.repo_root)
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)

        # === 导入 RAFT 和 InputPadder ===
        try:
            from core.raft import RAFT  # type: ignore
        except ImportError:
            try:
                from raft import RAFT  # type: ignore
            except ImportError as e2:
                raise ImportError(
                    "[RAFTOpticalFlow] Failed to import RAFT from repo_root. "
                    "Tried 'from core.raft import RAFT' and 'from raft import RAFT'."
                ) from e2

        try:
            from utils.utils import InputPadder  # type: ignore
        except Exception:
            InputPadder = None  # 没有也能跑，只是需要自己保证尺寸

        self.InputPadder = InputPadder

        # === 构建 args：既支持 'small' in args，又支持 args.small ===
        args = RAFTArgs(
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr,
            dropout=0.0,
        )

        # === 构建模型 ===
        base_model = RAFT(args)

        # === 解析权重路径并加载 ===
        weights_path = weights
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(self.repo_root, weights_path)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"[RAFTOpticalFlow] weight file not found: {weights_path}")

        state_dict = torch.load(weights_path, map_location="cpu")
        # Some RAFT releases wrap the actual weights in ``state_dict``. Keep
        # the adapter tolerant of both layouts and normalize DataParallel
        # prefixes before loading into the plain model. Constructing
        # ``torch.nn.DataParallel`` on CPU raises an IndexError and is also
        # unnecessary for this per-video inference worker.
        if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"[RAFTOpticalFlow] checkpoint must contain a state dict: {weights_path}"
            )
        if state_dict and all(str(key).startswith("module.") for key in state_dict):
            state_dict = {
                str(key)[len("module."):]: value for key, value in state_dict.items()
            }
        base_model.load_state_dict(state_dict)

        model = base_model.to(self.device)
        model.eval()

        self.model = model

    # ==================================================================
    # 对一串帧做相邻帧光流，返回每 pair 的平均光流幅值
    # ==================================================================
    @torch.no_grad()
    def compute_flow_for_frame_paths(
        self,
        frame_paths: List[str],
        iters: int = 20,
        resize_short_side: Optional[int] = None,
    ) -> List[float]:
        """
        :param frame_paths: 按时间顺序的帧路径列表
        :param iters:       RAFT 迭代次数
        :param resize_short_side: 为加速，可将短边缩放到该值（保持长宽比），None 表示不缩放
        :return: len(frame_paths) - 1 个 mean_mag
        """
        n = len(frame_paths)
        if n < 2:
            return []

        mag_list: List[float] = []

        for i in range(n - 1):
            f1 = frame_paths[i]
            f2 = frame_paths[i + 1]

            img1 = self._load_image(f1, resize_short_side)
            img2 = self._load_image(f2, resize_short_side)

            if img1 is None or img2 is None:
                mag_list.append(0.0)
                continue

            # (H, W, 3) -> (1, 3, H, W)
            image1 = torch.from_numpy(img1).permute(2, 0, 1)[None].float().to(self.device)
            image2 = torch.from_numpy(img2).permute(2, 0, 1)[None].float().to(self.device)

            if self.InputPadder is not None:
                padder = self.InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

            try:
                # 这里直接调用「单卡 RAFT 实例」，不经过 DataParallel
                _, flow_up = self.model(image1, image2, iters=iters, test_mode=True)

                # flow_up: (1, 2, H, W)
                flow = flow_up[0].detach().cpu().numpy()
                u, v = flow[0], flow[1]
                mag = np.sqrt(u * u + v * v).mean().item()
                if not np.isfinite(mag):
                    print(
                        f"[RAFT WARN] Non-finite flow magnitude on pair ({f1}, {f2}); "
                        "using 0.0."
                    )
                    mag = 0.0
            except Exception as e:
                print(
                    f"[RAFT WARN] Flow computation failed on pair ({f1}, {f2}): {repr(e)}"
                )
                mag = 0.0

            mag_list.append(float(mag))

        return mag_list

    # ==================================================================
    # 内部辅助函数
    # ==================================================================
    @staticmethod
    def _load_image(path: str, resize_short_side: Optional[int]) -> Optional[np.ndarray]:
        if not os.path.isfile(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if resize_short_side is not None and resize_short_side > 0:
            img = RAFTOpticalFlow._resize_short_side(img, resize_short_side)

        return img

    @staticmethod
    def _resize_short_side(img: np.ndarray, short_side: int) -> np.ndarray:
        h, w = img.shape[:2]
        if min(h, w) <= short_side:
            return img

        if h < w:
            new_h = short_side
            new_w = int(w * (short_side / h))
        else:
            new_w = short_side
            new_h = int(h * (short_side / w))

        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
