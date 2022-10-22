import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
# from transformers import ViTFeatureExtractor

import numpy as np
from scipy.spatial.transform import Rotation as R


def get_poses(scene, root):
    """
    :param scene: Index of trajectory
    :param root: Root folder of dataset
    :return: all camera poses as quaternion vector and 4x4 projection matrix
    """
    locations = []
    rotations = []
    loc_reader = open(root + '/SavedPosition_' + scene + '.txt', 'r')
    rot_reader = open(root + '/SavedRotationQuaternion_' + scene + '.txt', 'r')
    for line in loc_reader:
        locations.append(list(map(float, line.split())))

    for line in rot_reader:
        rotations.append(list(map(float, line.split())))

    locations = np.array(locations)
    rotations = np.array(rotations)
    # poses = np.concatenate([locations, rotations], 1)

    r = R.from_quat(rotations).as_matrix()

    TM = np.eye(4)
    TM[1, 1] = -1

    poses_mat = []
    for i in range(locations.shape[0]):
        ri = r[i]
        Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
        Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
        Pi_left = TM @ Pi @ TM   # Translate between left and right handed systems
        poses_mat.append(Pi_left)

    # return poses, np.array(poses_mat)
    return np.array(poses_mat)


def get_relative_pose(pose_t0, pose_t1):
    """
    :param pose_tx: 4x4 camera pose describing camera to world frame projection of camera x.
    :return: Position of camera 1's origin in camera 0's frame.
    """
    return np.matmul(np.linalg.inv(pose_t0), pose_t1)



class SimCol3DDataloader(Dataset):
    def __init__(self, subsets, subjects, ViT = False, transform=None):
        
        self.transform = transform
        self.frames_and_relpose = []
        self.ViT = ViT
        if self.ViT:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        
        for subset_id, subset in enumerate(subsets):
            for subject in subjects[subset_id]:

                Frame_list = np.sort(glob.glob(subset + '/Frames_' + subject + "/FrameBuffer*.png"))

                pose_loc = subset.split('Input')[0] +'GT/'
                poses = get_poses(subject, pose_loc)

                for frame_id in range(0,len(Frame_list)-1,1):
                    current_frame = Frame_list[frame_id]
                    next_frame = Frame_list[frame_id+1]

                    current_pose = poses[frame_id]
                    next_pose = poses[frame_id+1]
                    relative_pose = get_relative_pose(current_pose, next_pose)

                    self.frames_and_relpose.append([current_frame, next_frame, relative_pose])
        
    def __len__(self):
        return len(self.frames_and_relpose)

    def __getitem__(self, idx):
        
        # img
        if not self.ViT:
            curr_img = self.transform(Image.open(self.frames_and_relpose[idx][0]).convert('RGB'))
            next_img = self.transform(Image.open(self.frames_and_relpose[idx][1]).convert('RGB'))
        else:
            curr_img = self.feature_extractor(Image.open(self.frames_and_relpose[idx][0]).convert('RGB'), return_tensors="pt")
            next_img = self.feature_extractor(Image.open(self.frames_and_relpose[idx][1]).convert('RGB'), return_tensors="pt")
            curr_img = curr_img['pixel_values'].squeeze()
            next_img = next_img['pixel_values'].squeeze()
        relative_pose = self.frames_and_relpose[idx][2]
        relative_pose = relative_pose.flatten()
        relative_pose = relative_pose[:12]
        relative_pose = torch.from_numpy(relative_pose)
        relative_pose = relative_pose.type(torch.float32)
        return curr_img, next_img, relative_pose

