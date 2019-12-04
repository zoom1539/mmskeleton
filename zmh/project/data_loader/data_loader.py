import os
import json
import numpy as np
import torch

from . import pre_process


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_track = 1, repeat = 1, num_keypoints = -1):
        self.num_track = num_track
        self.num_keypoints = num_keypoints
        self.file_paths = [
            os.path.join(data_dir, name) for name in os.listdir(data_dir)
        ] * repeat
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        with open(self.file_paths[index]) as f:
            data = json.load(f)

        info = data['info']

        num_channel = len(info['keypoint_channels'])
        num_frame = info['num_frame']
        num_keypoints = info['num_keypoints'] if self.num_keypoints <= 0 else self.num_keypoints

        data['data'] = np.zeros(
            (self.num_track, num_channel, num_frame, num_keypoints)
        )

        annotations = data['annotations']

        for annotation in annotations:
            person_id = annotation['id']
            frame_index = annotation['frame_index']
            # if person_id < self.num_track and 
