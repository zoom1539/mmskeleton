import argparse
import os
import json

mode = dict(
    train = 'zmh/dataset/train',
    test = 'zmh/dataset/test',
    val = 'zmh/dataset/val')

def read_filename_list(work_dir):
    file_dir = os.path.join(work_dir, 'video')

    file_names = os.listdir(file_dir)
 
    return file_names


def generate_annotations(file_names, work_dir):
    filenames_no_ext = [name.split('.')[0] for name in file_names]

    # categories
    action_names = [name[-8:-4] for name in filenames_no_ext]
    categories = list(set(action_names))
    categories.sort()

    # annotations
    annotations = dict()
    for file_name, action_name in zip(file_names, action_names):
        annotations[file_name] = dict(category_id = categories.index(action_name))

    # json dump
    video_info = dict(categories = categories, annotations = annotations)
    with open(os.path.join(work_dir, 'annotation.json'), 'w') as f:
            json.dump(video_info, f, sort_keys=True, indent=4)
            # json.dump(video_info, f)


def main():
    parser = argparse.ArgumentParser(description='create_annotation.')
    parser.add_argument('mode', help='create mode')
    args = parser.parse_args()

    work_dir = mode[args.mode]
    file_names = read_filename_list(work_dir)
    generate_annotations(file_names, work_dir)


if __name__ == "__main__":
    main()