from queue import Queue
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import mmcv
from mmcv import ProgressBar
from mmcv.parallel import MMDataParallel
from mmskeleton.models.backbones import ST_GCN_18
from mmskeleton.utils import load_checkpoint
from zmh.processor import init_pose_estimator, inference_pose_estimator

video_paths = [ '/media/ubuntu/share/ntu_rgbd/nturgb+d_rgb/S010C001P007R001A024_rgb.avi',
                '/media/ubuntu/share/ntu_rgbd/nturgb+d_rgb/S010C001P007R001A024_rgb.avi',
                '/media/ubuntu/share/ntu_rgbd/nturgb+d_rgb/S010C001P007R001A028_rgb.avi',
                '/media/ubuntu/share/ntu_rgbd/nturgb+d_rgb/S010C001P007R001A024_rgb.avi',
                '/media/ubuntu/share/ntu_rgbd/nturgb+d_rgb/S010C001P007R001A043_rgb.avi',
                '/media/ubuntu/share/ntu_rgbd/nturgb+d_rgb/S010C001P007R001A024_rgb.avi',
                ]

video_path_result = 'zmh/work_dir/test1_concat.avi'

def generate_video(result_frames, fps, resolution):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  
    videoWriter = cv2.VideoWriter(video_path_result, fourcc, fps, resolution)  
    for frame in result_frames:
        videoWriter.write(frame)
    videoWriter.release()


def main():
    frames_total =[]
    for path in video_paths:
        frames = mmcv.VideoReader(path)
        frame_list = frames[:]
        frames_total = frames_total + frame_list

    generate_video(frames_total, frames.fps, frames.resolution)

if __name__ == "__main__":
    main()