from queue import Queue
import cv2
import mmcv
from zmh.processor import init_pose_estimator, inference_pose_estimator

video_path = 'zmh/work_dir/test.avi'
video_result_path = 'zmh/work_dir/test_result.avi'
recognize_frame_num = 60
gpu = 0
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

def read_video(video_path):
    reader = mmcv.VideoReader(video_path)
    return reader

def pose_estimate(pose_estimator, frame, id):
    pose = inference_pose_estimator(pose_estimator, frame)
    pose['frame_index'] = id

    return pose

def prepare_data(frame_queue):




    return data

def predict(data):
    pass

def render_frame(frame, pose, action_name):
    cv2.putText(frame, 
                action_name, 
                (100,100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2,
                (255,0,0), 
                5, 
                cv2.LINE_AA)
    return frame

def generate_video(result_frames):
    fps = 30   
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  
    videoWriter = cv2.VideoWriter(video_result_path, fourcc, fps, (1920,1080))  
    for frame in result_frames:
        videoWriter.write(frame)
    videoWriter.release()


def main():
    frames = read_video(video_path)

    frame_queue = Queue(recognize_frame_num)
    result_frames = []

    pose_estimator = init_pose_estimator( detection_cfg, estimation_cfg, device = gpu)

    for i, frame in enumerate(frames):
        pose = pose_estimate(pose_estimator, frame, i)

        if frame_queue.qsize() < recognize_frame_num:
            
            frame_queue.put(pose)
            result_frames.append(frame)
            continue
        else:
            frame_queue.get()
            frame_queue.put(pose)
            data = prepare_data(frame_queue)
            action_name = predict(data)
            frame = render_frame(frame, pose, action_name)
            result_frames.append(frame)
    
    generate_video(result_frames)

if __name__ == "__main__":
    main()