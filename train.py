import os.path
import os.path
import time

import torch
import torchvision
from torch.utils.data import DataLoader

import wandb
from dataset import dataset_factory
from frcnn.coco_eval import CocoEvaluator
from frcnn.coco_utils import convert_to_coco_api
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils import misc, wb_utils
from utils.chain_operators import SplittingOperator, ImgRescaleOperator, \
    SplitRegionOperator, LetterDetectionOperator, FinalOperator
from utils.coco_summary import summarizeCustom
from utils.misc import EarlyStop, display_terminal, display_terminal_eval, LossLoging

cpu_device = torch.device("cpu")


class Trainer:
    def __init__(self, args, fold=1, k_fold=5):
        device = torch.device('cuda' if args.cuda else 'cpu')
        self.args = args
        self._working_dir = os.path.join(args.checkpoints_dir, args.name, f'fold_{fold}')
        os.makedirs(self._working_dir, exist_ok=True)
        self._model = ModelsFactory.get_model(args, args.mode, self._working_dir, is_train=True, device=device,
                                              dropout=args.dropout)

        dataset_train = dataset_factory.get_dataset(args.dataset, args.mode, is_training=True,
                                                    image_size_p1=args.image_size, image_size_p2=args.p2_image_size,
                                                    ref_box_size=args.ref_box_height, fold=fold,
                                                    k_fold=k_fold, all_data=args.all_data)
        self.data_loader_train = DataLoader(dataset_train, shuffle=True, num_workers=args.n_threads_train,
                                            collate_fn=misc.collate_fn, persistent_workers=True,
                                            batch_size=args.batch_size, drop_last=True, pin_memory=True)
        dataset_val = dataset_factory.get_dataset(args.dataset, args.mode, is_training=False,
                                                  image_size_p1=args.image_size, image_size_p2=args.p2_image_size,
                                                  ref_box_size=args.ref_box_height, fold=fold, k_fold=k_fold)

        self.data_loader_val = DataLoader(dataset_val, shuffle=True, num_workers=args.n_threads_test,
                                          persistent_workers=True, pin_memory=True,
                                          collate_fn=misc.collate_fn, batch_size=args.batch_size)

        self.early_stop = EarlyStop(args.early_stop)
        print("Training sets: {} images".format(len(dataset_train)))
        print("Validating sets: {} images".format(len(dataset_val)))

        self._current_step = 0

    def is_trained(self):
        return self._model.existing()

    def set_current_step(self, step):
        self._current_step = step

    def load_pretrained_model(self):
        self._model.load()

    def train(self):
        best_m_ap = 0.

        for i_epoch in range(1, self.args.nepochs + 1):
            epoch_start_time = time.time()
            self._model.get_current_lr()
            # train epoch
            self._train_epoch(i_epoch)
            if self.args.lr_policy == 'step':
                self._model.lr_scheduler.step()

            if not i_epoch % self.args.n_epochs_per_eval == 0:
                continue

            val_dict, _ = self.validate(i_epoch, self.data_loader_val)
            current_m_ap = val_dict['val/mAP_0.5:0.95']
            if current_m_ap > best_m_ap:
                print("mAP_0.5:0.95 improved, from {:.4f} to {:.4f}".format(best_m_ap, current_m_ap))
                best_m_ap = current_m_ap
                for key in val_dict:
                    wandb.run.summary[f'best_model/{key}'] = val_dict[key]
                self._model.save()  # save best model

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self.args.nepochs, time_epoch, time_epoch / 60, time_epoch / 3600))

            if self.early_stop.should_stop(1 - current_m_ap):
                print(f'Early stop at epoch {i_epoch}')
                break

    def final_eval(self):
        val_dict, log_imgs = self.validate(self.args.nepochs + 1, self.data_loader_val, log_predictions=True)
        for key in val_dict:
            wandb.run.summary[f'best_model/{key}'] = val_dict[key]
        wandb.log({'val/all_predictions': log_imgs}, step=self._current_step)

    def _train_epoch(self, i_epoch):
        self._model.set_train()
        losses = LossLoging()
        for i_train_batch, train_batch in enumerate(self.data_loader_train):
            iter_start_time = time.time()

            train_loss = self._model.compute_loss(train_batch)
            self._model.optimise_params(train_loss)
            losses.update(train_loss)

            # update epoch info
            self._current_step += 1

            if self._current_step % self.args.save_freq_iter == 0:
                save_dict = losses.get_report()
                losses.clear()
                wandb.log(save_dict, step=self._current_step)
                display_terminal(iter_start_time, i_epoch, i_train_batch, len(self.data_loader_train), save_dict)

    def validate(self, i_epoch, val_loader, mode='val', log_predictions=False, max_dets=10000):
        self._model.set_eval()

        val_start_time = time.time()
        coco = convert_to_coco_api(val_loader.dataset)
        coco_evaluator = CocoEvaluator(coco, ["bbox"])
        coco_evaluator.coco_eval['bbox'].params.maxDets = [max_dets]
        coco_evaluator.coco_eval['bbox'].summarize = lambda: summarizeCustom(coco_evaluator.coco_eval['bbox'])

        if self.args.mode == 'first_twin':
            val_fn = self.second_twin_validate
        elif self.args.mode == 'second_twin':
            val_fn = self.first_twin_validation
        else:
            raise Exception(f'Train mode {self.args.mode} is not implemented!')

        coco_evaluator, logging_imgs = val_fn(val_loader, coco_evaluator, log_predictions)

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        coco_eval = coco_evaluator.coco_eval['bbox'].stats

        val_dict = {
            f'{mode}/mAP_0.5:0.95': coco_eval[0],
            f'{mode}/mAP_0.5': coco_eval[1],
            f'{mode}/mAP_0.75': coco_eval[2],
        }
        wandb.log(val_dict, step=self._current_step)
        display_terminal_eval(val_start_time, i_epoch, val_dict)

        return val_dict, logging_imgs

    def first_twin_validation(self, val_loader, coco_evaluator, log_predictions=False):

        to_pil_img = torchvision.transforms.ToPILImage()
        logging_imgs = []

        # NOTE: The operators below should be read from bottom up

        # Operators for localising letters inside each papyrus regions
        predictor = LetterDetectionOperator(FinalOperator(), self._model)
        predictor = SplittingOperator(predictor)
        predictor = SplitRegionOperator(predictor, self.args.p2_image_size, self.args.merge_iou_threshold)
        predictor = ImgRescaleOperator(predictor, self.args.ref_box_height)

        for image, target in val_loader.dataset:
            pil_img = to_pil_img(image)
            if len(target['boxes']) == 0:
                continue
            box_height = (target['boxes'][:, 3] - target['boxes'][:, 1]).mean()

            # We will use the regions and box_height from GT to evaluate for now
            # These information should be predicted by the p1 network
            predictions = predictor((pil_img, {'box_height': box_height}))
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in [predictions]]
            res = {target["image_id"].item(): output for target, output in zip([target], outputs)}
            coco_evaluator.update(res)

            if log_predictions:
                for i in range(len(outputs)):
                    img = wb_utils.bounding_boxes(image, outputs[i]['boxes'].numpy(),
                                                  outputs[i]['labels'].type(torch.int64).numpy(),
                                                  outputs[i]['scores'].numpy())
                    logging_imgs.append(img)
        return coco_evaluator, logging_imgs

    def second_twin_validate(self, val_loader, coco_evaluator, log_predictions=False):

        logging_imgs = []
        for i_train_batch, batch in enumerate(val_loader):
            images, targets = batch
            region_predictions = self._model.forward(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in region_predictions]

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            coco_evaluator.update(res)

            if log_predictions:
                for i in range(len(outputs)):
                    img = wb_utils.bounding_boxes(images[i], outputs[i]['boxes'].numpy(),
                                                  outputs[i]['labels'].type(torch.int64).numpy(),
                                                  outputs[i]['scores'].numpy())
                    logging_imgs.append(img)

        return coco_evaluator, logging_imgs


if __name__ == "__main__":
    train_args = TrainOptions().parse()
    wandb.init(group=train_args.group,
               name=train_args.name,
               project=train_args.wb_project,
               entity=train_args.wb_entity,
               resume=train_args.resume,
               config=train_args,
               mode=train_args.wb_mode)

    trainer = Trainer(train_args)
    if trainer.is_trained():
        trainer.set_current_step(wandb.run.step)
        trainer.load_pretrained_model()

    if train_args.resume or not trainer.is_trained():
        trainer.train()

    trainer.load_pretrained_model()
    trainer.final_eval()
