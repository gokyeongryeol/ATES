# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy
import torch
from typing import Sequence
import time
from functools import partial
from pycocotools import mask
import numpy as np
import json
from tqdm import tqdm

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.registry import LOOPS
from mmengine.runner.loops import TestLoop, _parse_losses, _update_losses
from mmengine.runner.amp import autocast
from mmengine.analysis import get_model_complexity_info


@LOOPS.register_module()
class InferenceLoop(TestLoop):
    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        data_to_save = self.runner.data_to_save
        data_to_save['annotations'] = []

        if self.runner.conf_threshold_json.endswith('.json'):
            conf_threshold = json.load(open(self.runner.conf_threshold_json, 'r'))
        else:
            conf_threshold = {str(cat_dict['id']): 0.01 for cat_dict in data_to_save['categories']}

        # clear test loss
        self.test_loss.clear()
        ann_id_cnt = 1
        for idx, data_batch in enumerate(tqdm(self.dataloader)):
            # outputs, inference_time, flops = self.run_iter(idx, data_batch)
            outputs = self.run_iter(idx, data_batch)

            bboxes = outputs[0].pred_instances.bboxes.cpu().numpy()
            scores = outputs[0].pred_instances.scores.cpu().numpy()
            labels = outputs[0].pred_instances.labels.cpu().numpy()

            for box, score, cat_id in zip(bboxes, scores, labels):
                if float(score) >= conf_threshold[str(cat_id)]:
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                    image_id = data_batch["data_samples"][0][0].img_id if isinstance(data_batch["data_samples"][0], tuple) else data_batch["data_samples"][0].img_id
                    data_to_save['annotations'].append({
                        "id": ann_id_cnt,
                        "image_id": image_id,
                        "category_id": int(cat_id),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                        "area": (x2-x1)*(y2-y1),
                        "iscrowd": 0,
                    })
                    ann_id_cnt += 1

        return data_to_save

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        return outputs


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('conf_threshold_json', help='confidence threshold json file')
    parser.add_argument('save_json_file', help='txt file to save')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # add `DumpResults` dummy metric
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))

    # prepare data to save
    data_to_save = json.load(open(os.path.join(
        cfg._cfg_dict['test_dataloader']['dataset']['data_root'],
        cfg._cfg_dict['test_dataloader']['dataset']['ann_file'],
    ), 'r'))

    runner.data_to_save = data_to_save
    runner.conf_threshold_json = args.conf_threshold_json

    # start testing
    data_to_save = runner.test()

    with open(args.save_json_file, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)


if __name__ == '__main__':
    main()
