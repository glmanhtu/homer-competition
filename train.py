import os.path
import os.path
import time

import torch
import torchvision.transforms
from torch.utils.data import DataLoader

import wandb
from dataset.papyrus import PapyrusDataset
from frcnn.coco_eval import CocoEvaluator
from frcnn.coco_utils import convert_to_coco_api
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils import misc, wb_utils
from utils.misc import EarlyStop, display_terminal, display_terminal_eval, convert_region_target, MetricLogging
from utils.transforms import ToTensor, Compose, ImageTransformCompose, FixedImageResize, RandomCropImage, PaddingImage, \
    ComputeAvgBoxHeight, LongRectangleCrop

args = TrainOptions().parse()


wandb.init(group=args.group,
           name=args.name,
           project=args.wb_project,
           entity=args.wb_entity,
           resume=args.resume,
           config=args,
           mode=args.wb_mode)


class Trainer:
    def __init__(self):
        device = torch.device('cuda' if args.cuda else 'cpu')

        self._working_dir = os.path.join(args.checkpoints_dir, args.name)
        self._model = ModelsFactory.get_model(args, self._working_dir, is_train=True, device=device,
                                              dropout=args.dropout)
        transforms = Compose([
            LongRectangleCrop(),
            RandomCropImage(min_factor=0.8, max_factor=1, min_iou_papyrus=0.2),
            PaddingImage(padding_size=50),
            FixedImageResize(args.image_size),
            ImageTransformCompose([
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                ], p=0.5)
            ]),
            ToTensor()
        ])
        dataset_train = PapyrusDataset(args.dataset, transforms, is_training=True)
        self.data_loader_train = DataLoader(dataset_train, shuffle=True, num_workers=args.n_threads_train,
                                            collate_fn=misc.collate_fn,
                                            batch_size=args.batch_size, drop_last=True, pin_memory=True)
        transforms = Compose([
            LongRectangleCrop(),
            PaddingImage(padding_size=50),
            FixedImageResize(args.image_size),
            ToTensor()])
        dataset_val = PapyrusDataset(args.dataset, transforms, is_training=False)

        self.data_loader_val = DataLoader(dataset_val, shuffle=True, num_workers=args.n_threads_test,
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
        for i_epoch in range(1, args.nepochs + 1):
            epoch_start_time = time.time()
            self._model.get_current_lr()
            # train epoch
            self._train_epoch(i_epoch)
            if args.lr_policy == 'step':
                self._model.lr_scheduler.step()

            if not i_epoch % args.n_epochs_per_eval == 0:
                continue

            val_dict, log_imgs = self._validate(i_epoch, self.data_loader_val)

            current_m_ap = val_dict['val/mAP_0.5:0.95']
            if current_m_ap > best_m_ap:
                print("mAP_0.5:0.95 improved, from {:.4f} to {:.4f}".format(best_m_ap, current_m_ap))
                best_m_ap = current_m_ap
                for key in val_dict:
                    wandb.run.summary[f'best_model/{key}'] = val_dict[key]
                self._model.save()  # save best model
                wandb.log({'val/prediction': log_imgs}, step=self._current_step)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, args.nepochs, time_epoch, time_epoch / 60, time_epoch / 3600))

            if self.early_stop.should_stop(1 - current_m_ap):
                print(f'Early stop at epoch {i_epoch}')
                break

        self.load_pretrained_model()
        _, log_imgs = self._validate(args.nepochs + 1, self.data_loader_val, log_first_img=False)
        wandb.log({'val/all_predictions': log_imgs}, step=self._current_step)

    def _train_epoch(self, i_epoch):
        self._model.set_train()
        losses = []
        for i_train_batch, train_batch in enumerate(self.data_loader_train):
            iter_start_time = time.time()

            train_loss = self._model.compute_loss(train_batch)
            self._model.optimise_params(train_loss)
            losses.append(train_loss.item())

            # update epoch info
            self._current_step += 1

            if self._current_step % args.save_freq_iter == 0:
                save_dict = {
                    'train/loss': sum(losses) / len(losses),
                }
                losses.clear()
                wandb.log(save_dict, step=self._current_step)
                display_terminal(iter_start_time, i_epoch, i_train_batch, len(self.data_loader_train), save_dict)

    @staticmethod
    def add_features(img_features, images, features):
        for image_name, features in zip(images, features):
            feature_cpu = features.cpu()
            if image_name not in img_features:
                img_features[image_name] = []
            img_features[image_name].append(feature_cpu)

    def _validate(self, i_epoch, val_loader, mode='val', log_first_img=True):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        cpu_device = torch.device("cpu")

        coco = convert_to_coco_api(val_loader.dataset, convert_region_target)
        iou_types = ["bbox"]
        coco_evaluator = CocoEvaluator(coco, iou_types)

        logging_imgs = []
        for i_train_batch, batch in enumerate(val_loader):
            images, target = batch
            region_predictions = self._model.forward(images)
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in region_predictions]
            region_target = [convert_region_target(x) for x in target]
            res = {target["image_id"].item(): output for target, output in zip(region_target, outputs)}
            coco_evaluator.update(res)

            for i in range(len(outputs)):
                img = wb_utils.bounding_boxes(images[i], outputs[i]['boxes'].numpy(),
                                              outputs[i]['labels'].type(torch.int64).numpy(),
                                              outputs[i]['scores'].numpy(), outputs[i]['extra_head_pred'],
                                              target[i]['avg_box_scale'].item())
                logging_imgs.append(img)
                if log_first_img:
                    break

        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        coco_eval = coco_evaluator.coco_eval['bbox'].stats

        val_dict = {
            f'{mode}/mAP_0.5:0.95': coco_eval[0],
            f'{mode}/mAP_0.5': coco_eval[1],
            f'{mode}/mAP_0.75': coco_eval[2],
            f'{mode}/mAP_0.5:0.95_small': coco_eval[3],
            f'{mode}/mAP_0.5:0.95_medium': coco_eval[4],
            f'{mode}/mAP_0.5:0.95_large': coco_eval[5],
        }
        wandb.log(val_dict, step=self._current_step)
        display_terminal_eval(val_start_time, i_epoch, val_dict)

        return val_dict, logging_imgs


if __name__ == "__main__":
    trainer = Trainer()
    if trainer.is_trained():
        trainer.set_current_step(wandb.run.step)
        trainer.load_pretrained_model()

    if args.resume or not trainer.is_trained():
        trainer.train()

    trainer.load_pretrained_model()
