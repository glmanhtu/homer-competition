import math
import time
import torch
from scipy.stats import pearsonr


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


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def split_sequence(sequence, chunk_size):
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i: i + chunk_size]


def map_location(cuda):
    if torch.cuda.is_available() and cuda:
        return torch.device('cuda')
    else:
        return torch.device('cpu')


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


class LossLoging:

    def __init__(self):
        self.losses = {}

    def update(self, all_losses):
        for key in all_losses.keys():
            self.losses.setdefault(key, []).append(all_losses[key].item())

    def get_report(self):
        result = {}
        for key in self.losses:
            result[key] = sum(self.losses[key]) / len(self.losses[key])
        return result

    def clear(self):
        self.losses = {}


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

    def get_report(self):
        result = {}
        scale_criterion = torch.nn.SmoothL1Loss()
        for key in self.actual:
            pred = torch.cat(self.predictions[key], dim=0).view(-1)
            actual = torch.cat(self.actual[key], dim=0).view(-1)
            pcc = pearsonr(pred.numpy(), actual.numpy())
            loss_scale = scale_criterion(pred, actual)
            result[f'{key}/loss'] = loss_scale
            result[f'{key}/pcc'] = pcc[0]
        return result


def split_region(width, height, size):
    n_rows = round(height / size)
    n_cols = round(width / size)
    return n_cols, n_rows


def add_items_to_group(items, groups):
    """
    Add list of items to groups,
    If there are no groups that match with the items, create a new group and put those item in this new group
    If there is only one matching group, add all these items to this group
    If there is more than one matching group, add all these items to the first group, then move items from
                other matching groups to this first group
    """
    reference_group = {}
    for g_id, group in enumerate(groups):
        for fragment_id in items:
            if fragment_id in group and g_id not in reference_group:
                reference_group[g_id] = group

    if len(reference_group) > 0:
        reference_ids = list(reference_group.keys())
        for fragment_id in items:
            reference_group[reference_ids[0]].add(fragment_id)
        for g_id in reference_ids[1:]:
            for fragment_id in reference_group[g_id]:
                reference_group[reference_ids[0]].add(fragment_id)
            del groups[g_id]
    else:
        groups.append(set(items))
