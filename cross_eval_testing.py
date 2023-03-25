import copy
import json
import os

import torch

import wandb
from dataset import dataset_factory
from options.cross_val_options import CrossValOptions
from testing import Predictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


args = CrossValOptions().parse()

if __name__ == "__main__":
    assert args.n_epochs_per_eval == args.nepochs, "In cross-validation, n_epochs_per_eval have to be equal to nepochs"
    for fold in range(args.k_fold):
        run = wandb.init(group=args.group,
                         name=f'{args.name}_fold-{fold}',
                         job_type="CrossEvalFinal",
                         project=args.wb_project,
                         entity=args.wb_entity,
                         resume=args.resume,
                         config=args,
                         settings=wandb.Settings(_disable_stats=True),
                         mode=args.wb_mode)

        dataset = dataset_factory.get_dataset(args.dataset, args.mode, is_training=False,
                                              image_size_p1=args.image_size, image_size_p2=args.p2_image_size,
                                              ref_box_size=args.ref_box_height, fold=fold, k_fold=args.k_fold)
        pretrained_rgd_dir = os.path.join(args.region_detection_model_dir, f"fold_{fold}")
        pretrained_ltd_dir = os.path.join(args.letter_detection_model_dir, f"fold_{fold}")
        net_predictor = Predictor(args, pretrained_rgd_dir, pretrained_ltd_dir,
                                  device=torch.device('cuda' if args.cuda else 'cpu'))

        predictions = net_predictor.predict_all(dataset)
        pred_ids = set([x['image_id'] for x in predictions['annotations']])
        gt = copy.deepcopy(dataset.data)

        for annotation in list(gt['annotations']):
            if annotation['image_id'] not in pred_ids:
                gt['annotations'].remove(annotation)
            else:
                annotation['iscrowd'] = 0

        with open("gt_tmp.json", "w") as outfile:
            json.dump(gt, outfile, indent=4)

        with open("pr_tmp.json", "w") as outfile:
            json.dump(predictions['annotations'], outfile, indent=4)

        cocoGt = COCO('gt_tmp.json')
        cocoDt = cocoGt.loadRes("pr_tmp.json")

        os.remove('gt_tmp.json')
        os.remove('pr_tmp.json')

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.maxDets = [10000]

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        val_dict = {
            f'val/mAP_0.5:0.95': cocoEval.stats[0],
            f'val/mAP_0.5': cocoEval.stats[1],
            f'val/mAP_0.75': cocoEval.stats[2],
        }

        cocoEval.params.useCats = False

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        val_dict.update({
            f'val/noCat/mAP_0.5:0.95': cocoEval.stats[0],
            f'val/noCat/mAP_0.5': cocoEval.stats[1],
            f'val/noCat/mAP_0.75': cocoEval.stats[2],
        })

        wandb.log(val_dict)
        print(val_dict)

        run.finish()
