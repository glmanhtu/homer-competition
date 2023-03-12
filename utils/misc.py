import time
import torch.nn.functional as F
import torch


class EarlyStop:

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.losses = []
        self.best_loss = 99999999

    def should_stop(self, loss):
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
        if len(self.losses) <= self.n_epochs:
            return False
        best_loss_pos = self.losses.index(self.best_loss)
        if len(self.losses) - best_loss_pos <= self.n_epochs:
            return False
        return True


def map_location(cuda):
    if torch.cuda.is_available() and cuda:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'


def display_terminal(iter_start_time, i_epoch, i_train_batch, num_batches, train_dict):
    t = (time.time() - iter_start_time)
    current_time = time.strftime("%H:%M", time.localtime(time.time()))
    output = "Time {}\tBatch Time {:.2f}\t Epoch [{}]([{}/{}])\t".format(current_time, t,
                                                                         i_epoch, i_train_batch, num_batches)
    for key in train_dict:
        output += '{} {:.4f}\t'.format(key, train_dict[key])
    print(output)


def display_terminal_eval(iter_start_time, i_epoch, eval_dict):
    t = (time.time() - iter_start_time)
    output = "\nEval Time {:.2f}\t Epoch [{}] \t".format(t, i_epoch)
    for key in eval_dict:
        output += '{} {:.4f}\t'.format(key, eval_dict[key])
    print(output + "\n")


def collate_fn(batch):
    return tuple(zip(*batch))


def convert_region_target(item):
    return {
        'boxes': item['regions'],
        'letter_boxes': item['boxes'],
        'labels': item['region_labels'],
        'iscrowd': torch.zeros(item['region_labels'].shape),
        'area': item['region_area'],
        'image_id': item['image_id']
    }


def filter_boxes(region_box, boxes):
    """
        get only the boxes that are lying inside the region_box
    """
    invalid_letter_boxes = torch.logical_or(boxes[:, 0] < region_box[0], boxes[:, 2] > region_box[2])
    invalid_letter_boxes = torch.logical_or(invalid_letter_boxes, boxes[:, 1] < region_box[1])
    invalid_letter_boxes = torch.logical_or(invalid_letter_boxes, boxes[:, 3] > region_box[3])
    return boxes[torch.logical_not(invalid_letter_boxes)]


def flatten(l):
    return [item for sublist in l for item in sublist]


class MetricLogging:

    def __init__(self):
        self.predictions = {}
        self.actual = {}

    def update(self, key, predicts, actual):
        if key not in self.predictions:
            self.predictions[key] = []
            self.actual[key] = []

        self.predictions[key].append(predicts.view(-1, 1))
        self.actual[key].append(actual.view(-1, 1))

    def get_raw_data(self, key):
        pred = torch.stack(self.predictions[key], dim=0)
        actual = torch.stack(self.actual[key], dim=0)
        return pred, actual

    def get_mae_loss(self, key):
        pred = torch.stack(self.predictions[key], dim=0)
        actual = torch.stack(self.actual[key], dim=0)
        return F.l1_loss(pred.view(-1), actual.view(-1))

    def get_mse_loss(self, key):
        pred = torch.stack(self.predictions[key], dim=0)
        actual = torch.stack(self.actual[key], dim=0)
        return F.mse_loss(pred.view(-1), actual.view(-1))
