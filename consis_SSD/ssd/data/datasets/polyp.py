import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from ssd.structures.container import Container

class PolypDataset(torch.utils.data.Dataset):
    class_names = ('__background__', 'polyp')
    def __init__(self, data_dir, split, transform=None, target_transform=None,):
        # as you would do normally
        self.data_dir = data_dir
        self.split = split # train, val, test
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        self.ids = PolypDataset._read_image_ids(image_sets_file) # 返回train, val, test中图片名称的列表
        

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        # load the image as a PIL Image
        image_id = self.ids[index]
        boxes, labels= self._get_annotation(image_id)
        image = self._read_image(image_id)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

            # left and right flip
            img_fliplr = np.fliplr(image)
            img_fliplr = np.ascontiguousarray(img_fliplr)

            # up and down flip
            img_flipud = np.flipud(image)
            img_flipud = np.ascontiguousarray(img_flipud)

            # lr and ud flip
            a = np.fliplr(image)
            img_fliplrud = np.flipud(a)
            img_fliplrud = np.ascontiguousarray(img_fliplrud)

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        # return the image, the targets and the index in your dataset
        return self.totensor(image), self.totensor(img_fliplr), self.totensor(img_flipud), self.totensor(img_fliplrud), targets, index
    
    def totensor(self, image):
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)
    
    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids
    
    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        # is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            # is_difficult_str = obj.find('difficult').text
            # is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))
    
    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % img_id)
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image