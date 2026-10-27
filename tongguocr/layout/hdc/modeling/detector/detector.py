from .oriented_cascade_rcnn import OCROrientedRCNN
from .oriented_cascade_rcnn_infer import OCROrientedRCNN_Infer
from .hdc_detector import HDCDetector
from .hdc_detector_infer import HDCDetector_Infer

_DETECTION_META_ARCHITECTURES = {
    "OCROrientedRCNN": OCROrientedRCNN,
    "OCROrientedRCNN_Infer": OCROrientedRCNN_Infer,
    "HDCDetector": HDCDetector,
    "HDCDetector_Infer": HDCDetector_Infer
}


def build_detection_model(cfg):
    model_cfg = cfg.model.copy()
    meta_arch = _DETECTION_META_ARCHITECTURES[model_cfg.type]
    model_cfg.pop('type')
    return meta_arch(**model_cfg)
