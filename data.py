import sys
sys.path.append("..")

import torch
import json
import cv2
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import utils_file
from scipy.ndimage import gaussian_filter

class KeyPoseDataset(Dataset):
    def __init__(self, img_path, annotate_path, transform=None):
        annotation_file = json.load(open(annotate_path, "rb"))
        self.annotations = annotation_file['annotations']
        self.img_path = img_path
        self.transform = transform
        
        self.input_width = 192
        self.input_height = 256
        
        self.heatmap_width = 48
        self.heatmap_height = 64
        self.no_keypoints = 17
        
        utils_file.log_info("Cleaning Database")
        self.annotations = self.clean_database(self.annotations)
        utils_file.log_info("Completed")
        utils_file.log_info(len(self.annotations))        
    
    def clean_database(self, annotations):
        final_annotation = []
        
        for a in annotations:
            if a["iscrowd"] != 0:
                continue
            
            keypt_array = np.asarray(a["keypoints"])
            keypts = np.reshape(keypt_array, (17,3))
            pt_validity = keypts[:, 2] > 0
            if sum(pt_validity) == 0:
                continue
            
            x_coord, y_coord, width, height = a["bbox"]
            if width < self.heatmap_width or height < self.heatmap_height:
                continue
            
            final_annotation.append(a)
            
        return final_annotation
            
    def __len__(self):
        return len(self.annotations)
    
    def get_image(self, annotations):
        
        x_start, y_start, box_w, box_h = annotations['bbox']
        img_id = str(annotations['image_id'])
        
        img_name = '000000000000'
        img_name = img_name[0:len(img_name) - len(img_id)] + img_id
        img = Image.open(self.img_path + '/' + img_name + '.jpg')
        
        rescaled_img = img.resize((self.input_width,self.input_height), box=(x_start, y_start, x_start+box_w, y_start+box_h))
        rescaled_img = np.asarray(rescaled_img)
        
        if len(rescaled_img.shape) != 3:
            rescaled_img = np.stack((rescaled_img,)*3, axis=-1)
            
        # preprocess = v2.Compose([
        #     v2.ToTensor(),
        #     v2.Lambda(lambda x: x / 255.0),
        #     v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ])
        
        # rescaled_img = preprocess(rescaled_img)
        
        mean=np.asarray([0.485, 0.456, 0.406])
        std=np.asarray([0.229, 0.224, 0.225])
        rescaled_img = rescaled_img.astype('float32')/255.0
        rescaled_img = (rescaled_img - mean) / std

        return torch.tensor(rescaled_img).permute(2,0,1).float()
    
    def get_ground_heatmap(self, annotations, gauss_sigma=2):
        
        #Parsing annotations
        x_start, y_start, box_w, box_h = annotations['bbox']
        keypoints_list = np.asarray(annotations['keypoints'])
        
        #x,y,v
        keypts = np.reshape(keypoints_list, (17,3))
        
        box_offset = np.asarray([x_start, y_start,0])
        box_dims = np.asarray([box_w,box_h,1])
        heatmap_dims = np.asarray([self.heatmap_width,self.heatmap_height,1])
        
        #rescale keypts
        keypoints = np.round((keypts - box_offset) * heatmap_dims / box_dims).astype(int)
        
        #generate ground-truth heatmaps
        gt_heatmap = np.zeros((self.no_keypoints,self.heatmap_height,self.heatmap_width))
        for j in range(self.no_keypoints):
            if keypoints[j,2] > 0: #plot valid pts
                y = keypoints[j,0]
                x = keypoints[j,1]
                
                # skip, if x or y are out of bound
                if x<0 or y<0 or x>=self.heatmap_height or y>=self.heatmap_width:
                    keypoints[j,2] = 0
                    continue 

                # set joint location in heatmap
                gt_heatmap[j,x,y] = 1.0

                # apply gaussian
                gt_heatmap[j,:,:] = gaussian_filter(gt_heatmap[j,:,:], sigma=gauss_sigma, mode='constant', cval=0.0)

                # normalize to 1
                gt_heatmap[j,:,:] = gt_heatmap[j,:,:] / np.max(gt_heatmap[j,:,:])
        
        # get validity vector
        gt_validity = keypoints[:,2]>0

        return torch.tensor(gt_heatmap).float(), torch.tensor(gt_validity).float()
            
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        input_img = self.get_image(annotation)
        heatmap, validity = self.get_ground_heatmap(annotation)
        
        return {
            'input_img': input_img, 
            'heatmap': heatmap,
            'validity': validity
        }


# Initialize Datasets

train_pose_data = KeyPoseDataset(img_path='C:/Object-Detection/coco2017/train2017', annotate_path='C:/Object-Detection/coco2017/annotations/person_keypoints_train2017.json')
train_loader = DataLoader(train_pose_data, batch_size=32, shuffle=True, num_workers=0)

val_pose_data = KeyPoseDataset(img_path='C:/Object-Detection/coco2017/val2017', annotate_path='C:/Object-Detection/coco2017/annotations/person_keypoints_val2017.json')
val_loader = DataLoader(val_pose_data, batch_size=32, shuffle=False, num_workers=0)


    