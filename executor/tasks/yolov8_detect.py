# executor/tasks/yolov8_detect.py
# -*- coding: utf-8 -*-
"""
YOLOv8 检测执行器（增强版）

兼容点：
- 保持原有输出格式：
  - run(frame_paths) -> Dict[str, List[Dict]]
    {
      "frame_path_1": [
        {"cls": int, "score": float, "bbox": [x1, y1, x2, y2]},
        ...
      ],
      ...
    }

增强点：
- 增加每帧的结构化特征构造：
  - build_frame_features(all_dets) -> Dict[str, Dict[str, float]]
    {
      "frame_path_1": {
        "num_objects": 3,
        "num_person": 1,
        "num_hand": 2,
        "num_screen": 0,
        "coverage_ratio": 0.32,
        "center_coverage_ratio": 0.18,
        "max_conf": 0.93,
        "mean_conf": 0.81,
        "class_entropy": 0.69,
      },
      ...
    }

- 提供 run_with_features(frame_paths)，一次性返回 (detections, features)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ultra_cfg_root = os.path.join(_repo_root, "model", "ultralytics")
os.makedirs(_ultra_cfg_root, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", _ultra_cfg_root)
from executor.tasks.ultralytics_compat import disable_runtime_installation
disable_runtime_installation()

try:
    from ultralytics import YOLO
except Exception as e:  # 环境里没装也不要在 import 阶段直接崩
    YOLO = None
    _yolo_import_error = e
else:
    _yolo_import_error = None
finally:
    disable_runtime_installation()


# ------------------------ 小工具函数 ------------------------


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s <= 0:
        return [0.0 for _ in xs]
    return [e / s for e in exps]


def _entropy(probs: List[float]) -> float:
    """简单熵，用来衡量类别多样性。"""
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs if p > 0.0)


# ------------------------ 配置与类型 ------------------------


@dataclass
class YoloConfig:
    weights: str = "models/yolo/yolov8m.pt"
    device: str = "cuda"  # or "cpu"
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    # 对常用类别的别名映射（可按自己数据集改）
    person_classes: Tuple[int, ...] = (0,)  # COCO: 0 -> person
    hand_classes: Tuple[int, ...] = ()     # 如果你有手的类别，就填进去
    screen_classes: Tuple[int, ...] = ()   # 显示器/屏幕等类别索引


DetDict = Dict[str, Any]


class YoloDetector:
    """
    YOLOv8 检测执行器。

    主要接口：
      - run(frame_paths) -> Dict[str, List[DetDict]]
      - build_frame_features(all_dets) -> Dict[str, Dict[str, float]]
      - run_with_features(frame_paths) -> (all_dets, features)
    """

    def __init__(self, config: Dict[str, Any]):
        self.raw_config = config

        models_conf = config.get("models", {}) if isinstance(config, dict) else {}
        weights = (
            models_conf.get("yolov8_weights")
            or models_conf.get("yolo")
            or "models/yolo/yolov8m.pt"
        )

        device = models_conf.get("yolo_device", "cuda")
        img_size = int(models_conf.get("yolo_img_size", 640))
        conf = float(models_conf.get("yolo_conf", 0.25))
        iou = float(models_conf.get("yolo_iou", 0.45))

        self.cfg = YoloConfig(
            weights=weights,
            device=device,
            img_size=img_size,
            conf_thres=conf,
            iou_thres=iou,
        )

        if YOLO is None:
            raise RuntimeError(
                f"ultralytics.YOLO not available: {_yolo_import_error}"
            )

        # 懒加载模型（也可以在这里直接加载）
        self._model: Optional[YOLO] = None
        self._class_names: Dict[int, str] = {}

    # -------------------- 内部工具 --------------------

    def _ensure_model(self) -> YOLO:
        if self._model is None:
            if not os.path.isfile(self.cfg.weights):
                raise FileNotFoundError(
                    f"YOLO weights must be an existing local file: {self.cfg.weights}. "
                    "Set models.yolo to downloaded weights before running the pipeline."
                )
            self._model = YOLO(self.cfg.weights)
            # names 可能是 dict 或 list
            names = getattr(self._model, "names", {})
            if isinstance(names, dict):
                self._class_names = {int(k): str(v) for k, v in names.items()}
            elif isinstance(names, (list, tuple)):
                self._class_names = {i: str(n) for i, n in enumerate(names)}
            else:
                self._class_names = {}
        return self._model

    # -------------------- 主检测接口 --------------------

    def run(self, frame_paths: List[str]) -> Dict[str, List[DetDict]]:
        """
        对一批帧进行检测。

        返回:
            all_dets: Dict[str, List[DetDict]]
        """
        if not frame_paths:
            return {}

        model = self._ensure_model()

        # ultralytics YOLO 支持直接传 list[str] 作为图片路径
        # 这里关闭多余输出，保证安静
        try:
            results = model(
                frame_paths,
                imgsz=self.cfg.img_size,
                conf=self.cfg.conf_thres,
                iou=self.cfg.iou_thres,
                verbose=False,
                device=self.cfg.device,
            )
        except Exception as e:
            msg = str(e).lower()
            if "torchvision::nms" in msg or "nms" in msg:
                results = model(
                    frame_paths,
                    imgsz=self.cfg.img_size,
                    conf=self.cfg.conf_thres,
                    iou=self.cfg.iou_thres,
                    verbose=False,
                    device="cpu",
                )
            else:
                raise

        all_dets: Dict[str, List[DetDict]] = {}

        for img_path, res in zip(frame_paths, results):
            det_list: List[DetDict] = []

            # res.boxes: [n, 6] (x1, y1, x2, y2, score, cls)
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                all_dets[img_path] = det_list
                continue

            # xyxy: (n, 4); conf: (n,); cls: (n,)
            xyxy = boxes.xyxy  # shape [n,4]
            conf = boxes.conf  # shape [n]
            cls = boxes.cls    # shape [n]

            if xyxy is None or conf is None or cls is None:
                all_dets[img_path] = det_list
                continue

            xyxy = xyxy.cpu().numpy()
            conf = conf.cpu().numpy()
            cls = cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                det_list.append(
                    {
                        "cls": int(cls[i]),
                        "score": float(conf[i]),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        # 可选加一个人类可读类别名，方便后面用：
                        "label": self._class_names.get(int(cls[i]), str(int(cls[i]))),
                    }
                )

            all_dets[img_path] = det_list

        return all_dets

    # ------------------ 特征构造接口 ------------------

    def build_frame_features(
        self,
        all_dets: Dict[str, List[DetDict]],
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        从 YOLO 检测结果构造每帧的特征。

        参数:
          - all_dets: run(frame_paths) 的返回
          - img_size: (W, H)，如果没有就用 bbox 范围做相对归一化（不精准但够用）

        返回:
          frame_features: Dict[frame_path, Dict[str, float]]
        """
        features: Dict[str, Dict[str, float]] = {}

        # 如果没传 img_size，就在所有 bbox 里扫一遍估个 max_x, max_y 做归一化
        est_w, est_h = None, None
        if img_size is None:
            max_x, max_y = 0.0, 0.0
            for det_list in all_dets.values():
                for d in det_list:
                    x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
                    max_x = max(max_x, x2)
                    max_y = max(max_y, y2)
            if max_x > 0 and max_y > 0:
                est_w, est_h = max_x, max_y

        W, H = img_size if img_size is not None else (est_w, est_h)
        if W is None or H is None:
            # 没有任何 bbox 时，退化成不计算面积类特征
            W, H = 1.0, 1.0

        img_area = float(W * H)
        center_x1, center_y1 = 0.25 * W, 0.25 * H
        center_x2, center_y2 = 0.75 * W, 0.75 * H

        for img_path, det_list in all_dets.items():
            num_objects = len(det_list)
            if num_objects == 0:
                features[img_path] = {
                    "num_objects": 0.0,
                    "num_person": 0.0,
                    "num_hand": 0.0,
                    "num_screen": 0.0,
                    "coverage_ratio": 0.0,
                    "center_coverage_ratio": 0.0,
                    "max_conf": 0.0,
                    "mean_conf": 0.0,
                    "class_entropy": 0.0,
                }
                continue

            num_person = 0
            num_hand = 0
            num_screen = 0

            total_area = 0.0
            center_area = 0.0

            confs: List[float] = []
            class_hist: Dict[int, int] = {}

            for d in det_list:
                cls = int(d.get("cls", -1))
                score = float(d.get("score", 0.0))
                x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])

                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                area = w * h

                total_area += area
                confs.append(score)

                class_hist[cls] = class_hist.get(cls, 0) + 1

                # 类别计数
                if cls in self.cfg.person_classes:
                    num_person += 1
                if cls in self.cfg.hand_classes:
                    num_hand += 1
                if cls in self.cfg.screen_classes:
                    num_screen += 1

                # 计算中心区域覆盖面积
                cx1 = max(x1, center_x1)
                cy1 = max(y1, center_y1)
                cx2 = min(x2, center_x2)
                cy2 = min(y2, center_y2)
                cw = max(0.0, cx2 - cx1)
                ch = max(0.0, cy2 - cy1)
                center_area += cw * ch

            coverage_ratio = total_area / img_area
            center_coverage_ratio = center_area / img_area

            max_conf = max(confs) if confs else 0.0
            mean_conf = float(sum(confs) / len(confs)) if confs else 0.0

            # 类别熵（先用频次 / 总数作 softmax，再算熵）
            counts = list(class_hist.values())
            probs = _softmax([float(c) for c in counts])
            class_entropy = _entropy(probs)

            features[img_path] = {
                "num_objects": float(num_objects),
                "num_person": float(num_person),
                "num_hand": float(num_hand),
                "num_screen": float(num_screen),
                "coverage_ratio": float(coverage_ratio),
                "center_coverage_ratio": float(center_coverage_ratio),
                "max_conf": float(max_conf),
                "mean_conf": float(mean_conf),
                "class_entropy": float(class_entropy),
            }

        return features

    # ------------------ 方便调用的封装 ------------------

    def run_with_features(
        self, frame_paths: List[str], img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[Dict[str, List[DetDict]], Dict[str, Dict[str, float]]]:
        """
        一次性返回 (detections, frame_features)。

        - 保持 run() 的兼容：detections 的结构完全一样。
        - features 可直接喂给 keyframe_selector 里新的 det_score 逻辑。
        """
        all_dets = self.run(frame_paths)
        feats = self.build_frame_features(all_dets, img_size=img_size)
        return all_dets, feats
