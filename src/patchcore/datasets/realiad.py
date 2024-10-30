import os
import json
from enum import Enum

import PIL
import torch
from torchvision import transforms

_CLASSNAMES = [
    "audiojack",
    "bottle_cap",
    "button_battery",
    "end_cap",
    "eraser",
    "fire_hood",
    "mint",
    "mounts",
    "pcb",
    "phone_battery",
    "plastic_nut",
    "plastic_plug",
    "porcelain_doll",
    "regulator",
    "rolled_strip_base",
    "sim_card_set",
    "switch",
    "tape",
    "terminalblock",
    "toothbrush",
    "toy",
    "toy_brick",
    "transistor1",
    "u_block",
    "usb",
    "usb_adaptor",
    "vcpill",
    "wooden_beads",
    "woodstick",
    "zipper"
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Real_IAD_Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for RealIAD.
    """

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the Real-IAD data folder.
            classname: [str or None]. Name of Real-IAD class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "OK"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            with open(os.path.join(self.source, "realiad_jsons", classname+".json"), 'r') as class_file:
                class_data = json.load(class_file)

            classpath = os.path.join(self.source, classname)
            anomaly_types = set()

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            if self.split == DatasetSplit.TRAIN or self.split == DatasetSplit.VAL:
                for item in class_data['train']:
                    if not item["anomaly_class"] in imgpaths_per_class[classname]:
                        imgpaths_per_class[classname][item['anomaly_class']] = []

                    imgpaths_per_class[classname][item['anomaly_class']].append(os.path.join(classpath, item['image_path']))
                    anomaly_types = anomaly_types | {item['anomaly_class']}
                
                for anomaly in anomaly_types:
                    if self.train_val_split < 1.0:
                        n_images = len(imgpaths_per_class[classname][anomaly])
                        train_val_split_idx = int(n_images * self.train_val_split)
                        if self.split == DatasetSplit.TRAIN:
                            imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                classname
                            ][anomaly][:train_val_split_idx]
                        elif self.split == DatasetSplit.VAL:
                            imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                classname
                            ][anomaly][train_val_split_idx:]

                
            
            if self.split == DatasetSplit.TEST:
                for item in class_data['test']:
                    if not item["anomaly_class"] in imgpaths_per_class[classname]:
                        imgpaths_per_class[classname][item['anomaly_class']] = []

                    if not item["anomaly_class"] in maskpaths_per_class[classname]:
                        maskpaths_per_class[classname][item['anomaly_class']] = []

                    imgpaths_per_class[classname][item['anomaly_class']].append(os.path.join(classpath, item['image_path']))
                    if item['mask_path'] != None:
                        maskpaths_per_class[classname][item['anomaly_class']].append(os.path.join(classpath, item['mask_path']))
                    else:
                        maskpaths_per_class[classname][item['anomaly_class']].append(None)

            # for anomaly in anomaly_types:
            #     anomaly_path = os.path.join(classpath, anomaly)
            #     anomaly_files = sorted(os.listdir(anomaly_path))
            #     imgpaths_per_class[classname][anomaly] = [
            #         os.path.join(anomaly_path, x) for x in anomaly_files
            #     ]

            #     if self.train_val_split < 1.0:
            #         n_images = len(imgpaths_per_class[classname][anomaly])
            #         train_val_split_idx = int(n_images * self.train_val_split)
            #         if self.split == DatasetSplit.TRAIN:
            #             imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
            #                 classname
            #             ][anomaly][:train_val_split_idx]
            #         elif self.split == DatasetSplit.VAL:
            #             imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
            #                 classname
            #             ][anomaly][train_val_split_idx:]

            #     if self.split == DatasetSplit.TEST and anomaly != "good":
            #         anomaly_mask_path = os.path.join(maskpath, anomaly)
            #         anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
            #         maskpaths_per_class[classname][anomaly] = [
            #             os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
            #         ]
            #     else:
            #         maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "OK":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
