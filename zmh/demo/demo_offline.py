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

video_path = 'zmh/work_dir/test_concat.avi'
video_result_path = 'zmh/work_dir/test_result.avi'
recognize_frame_num = 60
keypoints_num = 17
track_num = 1
batch_size = 1
channel_num = 3
gpu = 0
checkpoint = 'zmh/work_dir/stgcn.pth'

detection_cfg = dict(
    model_cfg='configs/mmdet/cascade_rcnn_r50_fpn_1x.py',
    checkpoint_file= 'mmskeleton://mmdet/cascade_rcnn_r50_fpn_20e',
    bbox_thre=0.8)

estimation_cfg = dict(
    model_cfg = 'configs/pose_estimation/hrnet/pose_hrnet_w32_256x192_test.yaml',
    checkpoint_file = 'mmskeleton://pose_estimation/pose_hrnet_w32_256x192',
    data_cfg = dict(
        image_size = [192, 256],
        pixel_std = 200,
        image_mean = [0.485, 0.456, 0.406],
        image_std = [0.229, 0.224, 0.225],
        post_process = True
    )
)

model_cfg = dict(
    in_channels = 3,
    num_class = 3,
    edge_importance_weighting = True,
    graph_cfg = dict(
        layout = "coco",
        strategy = "spatial",
    )
)

actions = ['kick', 'phone call', 'falling down']
    

def read_video(video_path):
    reader = mmcv.VideoReader(video_path)
    return reader

def pre_process(pose, resolution):
    if pose['has_return']:
        pose['joint_preds'][:, :, :] /= resolution
        pose['joint_preds'][:, :, :] -= 0.5

        mask = (pose['joint_scores'] == 0)
        mask = np.reshape(mask[0], pose['joint_scores'].shape[1])
        pose['joint_preds'][:, mask, :] = 0

    return pose

def pose_estimate(pose_estimator, frame, id, resolution):
    pose = inference_pose_estimator(pose_estimator, frame)
    pose['frame_index'] = id

    pose = pre_process(pose, resolution)

    return pose

def prepare_data(pose_queue):
    data= np.zeros((batch_size, channel_num, 
                    recognize_frame_num, keypoints_num, 
                    track_num),
                    dtype=np.float32)

    for i in range(recognize_frame_num):
        pose = pose_queue.get()
        data[0, 0:2, i, :, 0] = pose['joint_preds'][0].reshape(keypoints_num,2).transpose()
        data[0, 2, i, :, 0] = pose['joint_scores'][0].reshape(keypoints_num)
        pose_queue.put(pose)

    data = torch.from_numpy(data)
    return data

def build_model():
    model = ST_GCN_18(**model_cfg)

    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(1)).cuda()
    model.eval()

    return model
    

def predict(model, data):

    with torch.no_grad():
        output = model(data).data
        output = output[0]
        score = F.softmax(output, dim=0).cpu().numpy()

    max_id = np.argmax(score)

    action_name = ''
    if score[max_id] > 0.7:
        action_name = actions[max_id]
    return action_name, score[max_id]

def render_frame(frame, pose, action_name, score):
    cv2.putText(frame, 
                action_name, 
                (100,100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2,
                (255,0,0), 
                5, 
                cv2.LINE_AA)
    cv2.putText(frame, 
                str(score), 
                (100,200), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2,
                (255,0,0), 
                5, 
                cv2.LINE_AA)
    return frame

def generate_video(result_frames, resolution):
    fps = 30   
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  
    videoWriter = cv2.VideoWriter(video_result_path, fourcc, fps, resolution)  
    for frame in result_frames:
        videoWriter.write(frame)
    videoWriter.release()


def main():
    frames = read_video(video_path)

    pose_queue = Queue(recognize_frame_num)
    result_frames = []

    pose_estimator = init_pose_estimator( detection_cfg, estimation_cfg, device = gpu)
    model = build_model()

    prog_bar = ProgressBar(len(frames))

    for i, frame in enumerate(frames):
        prog_bar.update()

        pose = pose_estimate(pose_estimator, frame, i, frames.resolution)

        if pose_queue.qsize() < recognize_frame_num:

            pose_queue.put(pose)
            result_frames.append(frame)
            continue
        else:
            pose_queue.get()
            pose_queue.put(pose)
            data = prepare_data(pose_queue)
            action_name, score = predict(model, data)
            frame = render_frame(frame, pose, action_name, score)
            result_frames.append(frame)
        
    
    generate_video(result_frames, frames.resolution)

if __name__ == "__main__":
    main()