import os
import uuid
import zipfile
import shutil
from typing import Callable

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def read_emotion_labels(label_file):
    emotion_labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            image_name = parts[0]
            emotion_label = parts[1]
            emotion_labels[image_name] = emotion_label
    return emotion_labels


def process_dataset(image_dir, emotion_label_file, output_dir, image_name_trans: Callable[[str], str] = lambda s: s,
                    emotion_label_trans: Callable[[str], str] = lambda s: s):
    # Read emotion labels
    emotion_labels = read_emotion_labels(emotion_label_file)

    # Create dataset directory structure
    if os.path.exists(output_dir) and os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
        print('Output directory already exists.')
        if input('Delete it? [Y/N]').upper() not in ['Y', 'YES', '']:
            return
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'training')
    test_dir = os.path.join(output_dir, 'validation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move images and annotations to train/test directories
    for image_name, emotion_label in emotion_labels.items():
        image_path = os.path.join(image_dir, image_name_trans(image_name))
        if 'train' in image_name:
            dest_dir = train_dir
        elif 'test' in image_name:
            dest_dir = test_dir
        else:
            continue
        dest_dir = os.path.join(dest_dir, emotion_label_trans(emotion_label))  # Append image class dir
        os.makedirs(dest_dir, exist_ok=True)

        shutil.move(image_path, dest_dir)

    print('Dataset created successfully!')


emotion_map = {
    '1': 'surprise',
    '2': 'fear',
    '3': 'disgust',
    '4': 'happiness',
    '5': 'sadness',
    '6': 'anger',
    '7': 'neutral'
}

temp_image_dir = f'{project_root}/temp/{str(uuid.uuid1())}'
os.makedirs(temp_image_dir, exist_ok=True)
extract_zip(f'{project_root}/raw_data/basic/Image/aligned.zip', temp_image_dir)
process_dataset(image_dir=f'{temp_image_dir}/aligned',
                image_name_trans=lambda s: s.replace('.', '_aligned.'),
                emotion_label_trans=lambda s: emotion_map[s],
                emotion_label_file=f'{project_root}/raw_data/basic/EmoLabel/list_patition_label.txt',
                output_dir=f'{project_root}/data')
shutil.rmtree(temp_image_dir)
