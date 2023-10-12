# PapyTwin network

This is a PyTorch implementation of the network described in the paper "PapyTwin net: a Twin network for Greek letters detection on
ancient Papyri""


## Compatibility
The dependencies can be installed by running the following command:
```bash
pip install -r requirements.txt
```

# Training
## First twin:
```bash
python -u train.py --dataset /path/to/HomerCompTraining --batch_size 5 --name first_twin_res50_bs5 --n_epochs_per_eval 5 --cuda --lr 4e-3 --early_stop 10 --nepochs 300 --lr_policy step --lr_decay_epochs 100 --image_size 800 --p2_image_size 800 --mode first_twin --p1_arch resnet50
```
## Second twin:
```bash
python -u train.py --dataset /path/to/HomerCompTraining --batch_size 5 --name 2Twin_res50_box96_bs5 --n_epochs_per_eval 5 --cuda --lr 4e-3 --early_stop 50 --nepochs 500 --lr_policy step --lr_decay_epochs 30 --image_size 800 --p2_image_size 800 --mode second_twin --p1_arch resnet50 --p2_arch resnet50 --ref_box_height 96
```

# Reproducing the results
## Pretrained models
Link to download: https://mega.nz/folder/349AhR7Z#1y8CGz9YUiNoVlSv3uw7jQ

## Generate ensemble prediction results
```bash
python -u ensemble_testing.py --pretrained_model_path /path/to/pretrain_models --dataset /path/to/HomerCompTesting --n_threads_train 0 --n_threads_test 0 --image_size 800 --p2_image_size 800 --mode testing --name CV_2Twin_res50_box96 --ref_box_height 96 --cuda --p1_arch resnet50 --p2_arch resnet50 --merge_iou_threshold 0.7 --prediction_path predictions/CV_2Twin_res50_box96
```