# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import numpy as np
import numpy as np
import mmcv
import cv2


def is_2dlist(x):
    """check x is 2d-list([[1], []]) or 1d empty list([]).

    Notice:
        The reason that it contains 1d empty list is because
        some arguments from gt annotation file or model prediction
        may be empty, but usually, it should be 2d-list.
    """
    if not isinstance(x, list):
        return False
    if len(x) == 0:
        return True

    return all(isinstance(item, list) for item in x)


def is_type_list(x, type):

    if not isinstance(x, list):
        return False

    return all(isinstance(item, type) for item in x)


def is_none_or_type(x, type):

    return isinstance(x, type) or x is None


def equal_len(*argv):
    assert len(argv) > 0

    num_arg = len(argv[0])
    for arg in argv:
        if len(arg) != num_arg:
            return False
    return True


def valid_boundary(x, with_score=True):
    num = len(x)
    if num < 8:
        return False
    if num % 2 == 0 and (not with_score):
        return True
    if num % 2 == 1 and with_score:
        return True

    return False


def imshow_pred_boundary(img,
                         boundaries_with_scores,
                         labels,
                         score_thr=0,
                         boundary_color='blue',
                         text_color='blue',
                         thickness=1,
                         font_scale=0.5,
                         show=True,
                         win_name='',
                         wait_time=0,
                         out_file=None,
                         show_score=False):
    """Draw boundaries and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        boundaries_with_scores (list[list[float]]): Boundaries with scores.
        labels (list[int]): Labels of boundaries.
        score_thr (float): Minimum score of boundaries to be shown.
        boundary_color (str or tuple or :obj:`Color`): Color of boundaries.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename of the output.
        show_score (bool): Whether to show text instance score.
    """
    assert isinstance(img, (str, np.ndarray))
    assert is_2dlist(boundaries_with_scores)
    assert is_type_list(labels, int)
    assert equal_len(boundaries_with_scores, labels)
    if len(boundaries_with_scores) == 0:
        warnings.warn('0 text found in ' + out_file)
        return None

    valid_boundary(boundaries_with_scores[0])
    img = mmcv.imread(img)

    scores = np.array([b[-1] for b in boundaries_with_scores])
    inds = scores > score_thr
    boundaries = [boundaries_with_scores[i][:-1] for i in np.where(inds)[0]]
    scores = [scores[i] for i in np.where(inds)[0]]
    labels = [labels[i] for i in np.where(inds)[0]]

    boundary_color = mmcv.color_val(boundary_color)
    text_color = mmcv.color_val(text_color)
    font_scale = 0.5

    for boundary, score in zip(boundaries, scores):
        boundary_int = np.array(boundary).astype(np.int32)

        cv2.polylines(img, [boundary_int.reshape(-1, 1, 2)],
                      True,
                      color=boundary_color,
                      thickness=thickness)

        if show_score:
            label_text = f'{score:.02f}'
            cv2.putText(img, label_text,
                        (boundary_int[0], boundary_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


class TextDetectorMixin:
    """Base class for text detector, only to show results.

    Args:
        show_score (bool): Whether to show text instance score.
    """

    def __init__(self, show_score):
        self.show_score = show_score

    def show_result(self,
                    img,
                    result,
                    score_thr=0.5,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results to draw over `img`.
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.imshow_pred_boundary`
        """
        img = mmcv.imread(img)
        img = img.copy()
        boundaries = None
        labels = None
        if 'boundary_result' in result.keys():
            boundaries = result['boundary_result']
            labels = [0] * len(boundaries)

        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        if boundaries is not None:
            imshow_pred_boundary(img,
                                 boundaries,
                                 labels,
                                 score_thr=score_thr,
                                 boundary_color=bbox_color,
                                 text_color=text_color,
                                 thickness=thickness,
                                 font_scale=font_scale,
                                 win_name=win_name,
                                 show=show,
                                 wait_time=wait_time,
                                 out_file=out_file,
                                 show_score=self.show_score)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, '
                          'result image will be returned')
        return img
