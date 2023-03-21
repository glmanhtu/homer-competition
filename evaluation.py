import argparse
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def summarizeCustom(self):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    def _summarizeCustom():
        stats = np.zeros((1,))
        stats[0] = _summarize(1, maxDets=self.params.maxDets[0])
        return stats

    self.stats = _summarizeCustom()

COCOeval.summarizeCustom = summarizeCustom


def main():
    # Create the parser
    parser = argparse.ArgumentParser()  # Add an argument
    parser.add_argument('--prediction_file', type=str, required=True)  # Parse the argument
    parser.add_argument('--gt_file', type=str, required=True)  # Parse the argument
    args = parser.parse_args()

    jFile = open(os.path.join(args.prediction_file))
    predictions = json.load(jFile)
    jFile.close()

    pred_ids = set([x['image_id'] for x in predictions['annotations']])

    jFile = open(os.path.join(args.gt_file))
    gt = json.load(jFile)
    jFile.close()

    for annotation in list(gt['annotations']) :
        if annotation['tags']['BaseType'][0] == 'bt3' :
            gt['annotations'].remove(annotation)

        elif annotation['image_id'] not in pred_ids:
            gt['annotations'].remove(annotation)

    for annotation in gt['annotations']:
        annotation['iscrowd'] = 0

    with open("gt_tmp.json", "w") as outfile:
        json.dump(gt, outfile, indent=4)

    with open("pr_tmp.json", "w") as outfile:
        json.dump(predictions['annotations'], outfile, indent=4)


    cocoGt=COCO('gt_tmp.json')
    cocoDt=cocoGt.loadRes("pr_tmp.json")

    os.remove('gt_tmp.json')
    os.remove('pr_tmp.json')


    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.maxDets = [10000]
    

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarizeCustom()
    labelscore = cocoEval.stats[0]


    cocoEval.params.useCats = False

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarizeCustom()
    nolabelscore = cocoEval.stats[0]


    print("Score (no labels) : " + str(nolabelscore))
    print("Score (labels) : " + str(labelscore))


if __name__ == '__main__':
    main()

