import cv2
import numpy as np
from mmcv.image import imread, imwrite
from .color import color_val


def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] + 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)

def imshow_det_bboxes_w_uncert(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=2,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):
    """Draw bboxes and class labels (with scores and uncertainty) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores and uncertainty), shaped (n, 4) or
            (n, 5) or (n, 6).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]

    img = imread(img)
    if score_thr > 0:
        scores = bboxes[:, 4]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    # bbox_color = (179,112,117)
    bbox_color = (0,255,0)
    text_color = (255,255,255)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)

    
        if len(bbox) == 5:# (x1,y1,x2,y2,score)
            label_text += '|{:.02f}'.format(bbox[4])#score
        elif len(bbox) == 9:#(x1,y1,x2,y2,score,uncert_x1,uncert_y1,uncert_x2,uncert_y2)
            label_text += '|{:.02f}'.format(bbox[4])#score
            h = bbox[3]-bbox[1]
            w = bbox[2]-bbox[0]
            cv2.ellipse(img, (bbox_int[0], bbox_int[1]), (int(bbox[5]*w), int(bbox[6]*h)), 0., 0., 360, (180,120,31), 2*thickness)
            cv2.ellipse(img, (bbox_int[2], bbox_int[3]), (int(bbox[7]*w), int(bbox[8]*h)), 0., 0., 360, (180,120,31), 2*thickness)

        elif len(bbox) == 10:#(x1,y1,x2,y2,score,uncert_x1,uncert_y1,uncert_x2,uncert_y2,cent)
            label_text += ':{:.02f}'.format(bbox[4])#score
            # label_text += '|{:.02f}'.format(bbox[-1])#cent
            label_text_line2 = 'uncertainty:{:.02f}'.format(bbox[5:9].mean())#uncert
            h = bbox[3]-bbox[1]
            w = bbox[2]-bbox[0]
            cv2.ellipse(img, (bbox_int[0], bbox_int[1]), (int(bbox[5]*w), int(bbox[6]*h)), 0., 0., 360, (180,120,31), 2*thickness)
            cv2.ellipse(img, (bbox_int[2], bbox_int[3]), (int(bbox[7]*w), int(bbox[8]*h)), 0., 0., 360, (180,120,31), 2*thickness)

        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] + 10),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        cv2.putText(img, label_text_line2, (bbox_int[0], bbox_int[1] + 20),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
        # text_size, _ = cv2.getTextSize(label_text_line2, cv2.FONT_HERSHEY_COMPLEX, font_scale, thickness)
        # text_w, text_h = text_size
        # img_bg = cv2.rectangle(img.copy(), (bbox_int[0], bbox_int[1]), (bbox_int[0] + text_w, bbox_int[1] + text_h+10), (0,0,0), -1)
        # img = cv2.addWeighted(img_bg, 0.3, img, 0.7, 0, img)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        outer_path = out_file.split('/')[:-2]
        inner_path = out_file.split('/')[-2]
        out_file = out_file.replace('.png','.pdf')
        import os
        from PIL import Image
        out_path = os.path.join(*outer_path)
        in_path = os.path.join(out_path, inner_path)
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(in_path, exist_ok=True)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.save(out_file, "PDF", resolution=100.0)
        # imwrite(img, out_file)

    