
from myData import datasetA_fold,datasetB_fold, get_transform
from engine import train_one_epoch, evaluate, get_detections_on_image
import utils
from my_backbone_utils import resnet_fpn_backbone18,LinearC,resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN, FasterRCNN
#from my_fast_rcnn import FasterRCNN
import torch

def get_model_instance_segmentation(fold,type,ds):

    if "resnet" in type:
        backbone = resnet_fpn_backbone18(trainable_layers=5)
    else:
        backbone_dir = "/mnt/ssd/miccai/forestnets/results/models/"+ds+"1/"+type+"_100_"+fold+".pt"

        backbone = resnet_fpn_backbone(backbone_dir,trainable_layers=5)

    model = MaskRCNN(backbone, num_classes=2,rpn_score_thresh=0.1,rpn_nms_thresh=0.9,box_score_thresh=0.5)
    directory = "/mnt/ssd/miccai/forestnets/detection/results/models/"+ds+"5/maskRCNN/"
    model_dict_file = directory+type+"_"+fold+".pt"

    #print("=======",model_dict_file)
    model_dict = torch.load(model_dict_file)

    model.load_state_dict(model_dict)

    return model

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ds = "B"
    for extra in [""]:
        for fold in [3]:
            for type in ["original"]:

                if "A" in ds:
                    root="/mnt/ssd/brest_data/ultrasound/external/Dataset_BUSI_with_GT"
                    dataset_test = datasetA_fold(root,get_transform(train=False),"test",fold)
                else:
                    root="/mnt/ssd/brest_data/ultrasound/external/BUS"
                    dataset_test = datasetB_fold(root,get_transform(train=False),"test",fold)
                reduce_samples=False
                if reduce_samples:
                    n_samples = len(dataset_test)
                    indices = torch.randperm(n_samples).tolist()
                    dataset_test = torch.utils.data.Subset(dataset_test, indices[0:n_samples//5])


                data_loader_test = torch.utils.data.DataLoader(
                    dataset_test, batch_size=4, shuffle=False, num_workers=4,collate_fn=utils.collate_fn)

                # get the model using our helper function
                model = get_model_instance_segmentation(extra+str(fold),type,ds)

                # move model to the right device
                model.to(device)

                #evaluate(model, data_loader_test, device=device)

                get_detections_on_image(model, data_loader_test, ds, type, device=device)


           # print("That's it for fold "+str(fold)+"!")

main()