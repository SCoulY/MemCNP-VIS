
import os
import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet.core import results2json_videoseg, ovis_eval_vis
from mmdet import datasets
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def single_test(model, data_loader, show=False, save_path='', thresh=0.5):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES,
                                     save_vis = True,
                                     save_path = save_path,
                                     is_video = True,
                                     score_thr = thresh)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--save_vis_path', 
        default=None,
        type=str,
        help='path to save visual result')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', 
        help='output result file')
        
    parser.add_argument('--load_result', 
        default=False,
        type=str2bool, 
        help='whether to load existing result')

    parser.add_argument(
        '--eval',
        default=[],
        type=str,
        nargs='+',
        choices=['bbox', 'segm'],
        help='eval types')
    parser.add_argument('--show', 
        default=False,
        type=str2bool, 
        help='show results')
    parser.add_argument('--dump_bbox', 
        default=False,
        type=str2bool, 
        help='False when evaluating on YVIS')
    parser.add_argument(
        '--thresh', default=0.5, type=float, help='box score threshold')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    assert os.path.exists(args.config)
    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))

    assert args.gpus == 1
    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint)
    model = MMDataParallel(model, device_ids=[0])
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False)

    if args.load_result:
        outputs = mmcv.load(args.out)
    else:
        if args.save_vis_path:
            os.makedirs(args.save_vis_path, exist_ok=True)
        outputs = single_test(model, data_loader, args.show, save_path=args.save_vis_path, thresh=args.thresh)

    if args.out:
        if not args.load_result:
            print('writing results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            result_file = args.out + ('.json' if not args.out.endswith('.json') else '')
            if not args.load_result:
                results2json_videoseg(dataset, outputs, result_file, dump_bbox=args.dump_bbox)
            ovis_eval_vis(result_file, eval_types, dataset)

if __name__ == '__main__':
    main()
