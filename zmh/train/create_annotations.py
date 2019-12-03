import argparse
import os
import shutil
import json
from mmcv import ProgressBar


mode = dict(
    train = 'zmh/dataset/train',
    test = 'zmh/dataset/test',
    val = 'zmh/dataset/val')

categories = ['normal', 'A024', 'A028', 'A043']

def read_filename_list(work_dir):
    file_dir = os.path.join(work_dir, 'video')

    file_names = os.listdir(file_dir)
    file_names = [name for name in file_names if name != '.gitkeep']
    return file_names

def generate_annotations_sample(file_names, work_dir, sample_period):
    filenames_no_ext = [name.split('.')[0] for name in file_names]

    action_names = [name[-8:-4] for name in filenames_no_ext]

    # annotations
    annotations = dict()
    normal_count = 0

    prog_bar = ProgressBar(len(file_names))

    for file_name, action_name in zip(file_names, action_names):
        prog_bar.update()

        try:
            category_id = categories.index(action_name)
            annotations[file_name] = dict(category_id = category_id)
            file_path = os.path.join(work_dir, 'video', file_name)
            dst_path = os.path.join(work_dir, 'video_sampled', file_name)
            shutil.copyfile(file_path, dst_path)
        except ValueError:
            normal_count += 1
            category_id = 0
            if normal_count % sample_period == 0:
                annotations[file_name] = dict(category_id = category_id)
                file_path = os.path.join(work_dir, 'video', file_name)
                dst_path = os.path.join(work_dir, 'video_sampled', file_name)
                shutil.copyfile(file_path, dst_path)

    # json dump
    video_info = dict(categories = categories, annotations = annotations)
    with open(os.path.join(work_dir, 'annotation.json'), 'w') as f:
            json.dump(video_info, f, sort_keys=True, indent=4)
            # json.dump(video_info, f)



def generate_annotations(file_names, work_dir):
    mode = work_dir.split('/')[-1]

    if mode == 'train':
        sample_period = 6
    if mode == 'val':
        sample_period = 3
    if mode == 'test':
        sample_period = 2
    
    generate_annotations_sample(file_names, work_dir, sample_period)


def main():
    parser = argparse.ArgumentParser(description='create_annotation.')
    parser.add_argument('mode', help='create mode')
    args = parser.parse_args()

    work_dir = mode[args.mode]
    file_names = read_filename_list(work_dir)
    generate_annotations(file_names, work_dir)


if __name__ == "__main__":
    main()