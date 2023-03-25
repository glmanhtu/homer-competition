import numpy as np
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
        stats = np.zeros((3,))
        stats[0] = _summarize(1, maxDets=self.params.maxDets[0])
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[0])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[0])
        return stats

    self.stats = _summarizeCustom()


def summarize(cocoGt, cocoDt, max_dets=10000):
    COCOeval.summarizeCustom = summarizeCustom
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.maxDets = [max_dets]

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

    return val_dict
