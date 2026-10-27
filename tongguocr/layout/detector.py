from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from .connect import Connector
from .dewarp import Dewarper
from .hdc.modeling.detector.detector import build_detection_model
from .order import get_order


class LayoutInfer:
    """Minimal detector/dewarp/order wrapper for layout-only inference."""

    def __init__(self, config: dict[str, Any]):
        layout_cfg = config["layout_inference"]
        self.device = layout_cfg["device"]
        self.det_config_path = layout_cfg["detector_config_path"]
        self.det_checkpoint = layout_cfg["detector_checkpoint_path"]
        self.det_score_thr = float(layout_cfg["detector_score_threshold"])
        self.batch_size = int(layout_cfg["batch_size"])
        self.is_static = False

        self.dewarper = Dewarper()
        self.det_model = self.init_detector(self.det_config_path, self.det_checkpoint)
        self.connector = Connector(self.device, batch_size=self.batch_size, is_static=self.is_static)

    def init_detector(self, config: str | Path | mmcv.Config, checkpoint: str | Path | None = None):
        if isinstance(config, (str, Path)):
            config = mmcv.Config.fromfile(str(config))
        elif not isinstance(config, mmcv.Config):
            raise TypeError(f"config must be a filename or Config object, but got {type(config)}")

        if "pretrained" in config.model:
            config.model.pretrained = None
        elif "init_cfg" in config.model.backbone:
            config.model.backbone.init_cfg = None
        config.model.train_cfg = None
        model = build_detection_model(config)
        if checkpoint is not None:
            state = torch.load(str(checkpoint), map_location="cpu")
            model.load_state_dict(state)
        model.cfg = config
        model.to(self.device)
        return model.eval()

    def dewarp(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        img, grid, org_w, org_h = self.dewarper.dewarp(img)
        img_info = {
            "filename": image_path,
            "ori_filename": image_path,
            "img": img,
            "ori_shape": img.shape,
        }
        return img, img_info, grid, org_w, org_h

    def det_eval(self, img_info: dict[str, Any]):
        cfg = self.det_model.cfg
        device = next(self.det_model.parameters()).device
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline[1:])
        data = test_pipeline(img_info)
        data = collate([data], samples_per_gpu=1)
        data["img_metas"] = [img_metas.data[0] for img_metas in data["img_metas"]]
        data["img"] = [img.data[0] for img in data["img"]]
        if next(self.det_model.parameters()).is_cuda:
            data = scatter(data, [device])[0]

        det_polys = []
        det_segs = []
        with torch.no_grad():
            det_result = self.det_model(**data, rescale=True)
            instance_num = len(det_result[0]["seg_result"])
            for i in range(instance_num):
                poly = det_result[0]["bbox_result"][i]
                seg = det_result[0]["seg_result"][i]
                det_polys.append(poly)
                det_segs.append(seg)
        return det_polys, det_segs

    def det_recg_connect(self, img_raw, det_polys, det_segs):
        return self.connector(img_raw, det_polys, det_segs)

    def order_eval(self, points_batch, text_batch, width: int, height: int):
        return get_order(points_batch, text_batch, (height, width))
