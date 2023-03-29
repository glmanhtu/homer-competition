import csv
import os

import yaml

from dataset.papyrus import letter_mapping
from dataset.papyrus_p2 import PapyrusP2Dataset
from options.train_options import TrainOptions

args = TrainOptions(save_conf=False).parse()

train_dataset = PapyrusP2Dataset(args.dataset, is_training=True, image_size=args.p2_image_size,
                                 ref_box_size=args.ref_box_height)
test_dataset = PapyrusP2Dataset(args.dataset, is_training=False, image_size=args.p2_image_size,
                                ref_box_size=args.ref_box_height)


def generate_dataset(ds, out_path, label_path):
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    for image, target in ds:
        image_id = target['image_id'].item()
        n_cols, n_rows, col, row = tuple(target['image_part'].numpy())
        image_name = f'{image_id}-{n_cols}_{n_rows}_{col}_{row}'
        image.save(os.path.join(out_path, image_name + ".jpg"))

        im_w, im_h = image.width, image.height

        records = []
        for box, label in zip(target['boxes'], target['labels']):
            x_center = (box[2] - box[0]) / 2.
            y_center = (box[3] - box[1]) / 2.
            width = box[2] - box[0]
            height = box[3] - box[1]
            row = {
                'class': label.item() - 1,
                'x_center': x_center / im_w,
                'y_center': y_center / im_h,
                'width': width / im_w,
                'height': height / im_h
            }
            records.append(row)

        with open(os.path.join(label_path, image_name + '.txt'), 'w') as f:
            dict_writer = csv.DictWriter(f, fieldnames=records[0].keys())
            dict_writer.writerows(records)


generate_dataset(train_dataset, os.path.join(args.output_dataset, 'images', 'train'),
                 os.path.join(args.output_dataset, 'labels', 'train'))
generate_dataset(test_dataset, os.path.join(args.output_dataset, 'images', 'val'),
                 os.path.join(args.output_dataset, 'labels', 'val'))

config = {
    'path': args.output_dataset,
    'train': os.path.join(args.output_dataset, 'train'),
    'val': os.path.join(args.output_dataset, 'val'),
    'names': {v - 1: k for k, v in letter_mapping.items()}
}

with open(os.path.join(args.output_dataset, 'data.yaml'), "w") as file:
    yaml.dump(config, file, default_flow_style=False)
