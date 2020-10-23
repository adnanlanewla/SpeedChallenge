import os
import numpy as np

def get_filenames_labels(image_directory):
    filenames = os.listdir(image_directory)
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(image_directory, x)))
    labels = []
    sorted_filenames = []
    for file in filenames:
        if(file.endswith('.jpg') or file.endswith('.png')):
            image_file_name = os.path.splitext(file)[0]
            split_filename = image_file_name.split('_')
            label = split_filename[len(split_filename)-1]
            sorted_filenames.append(os.path.join(image_directory,file))
            labels.append(label)

    print(len(sorted_filenames))
    print(len(labels))

    return sorted_filenames, labels

def fileIO_for_opti_flow(image_directory):
    filenames = os.listdir(image_directory)
    filenames.sort(key=lambda x: os.path.getmtime(os.path.join(image_directory, x)))
    accelerations = []
    new_opti_flow_filenames = []
    speed_differences = open('../data/Speed_Differences.txt', 'r')

    for i in range(len(filenames) - 1):
        file1 = filenames[i]
        file2 = filenames[i + 1]
        if (file1.endswith('.jpg') and file1.endswith('.jpg')):
            image_file_name1 = os.path.splitext(file1)[0]
            split_filename1 = image_file_name1.split('_')

            image_file_name2 = os.path.splitext(file2)[0]
            split_filename2 = image_file_name2.split('_')

            acceleration = speed_differences.readline()
            acceleration = acceleration.rstrip()

            new_filename = split_filename1[0] + '_' + split_filename1[1] + '_' + split_filename1[2] + '_' + \
                           split_filename1[3] + '_' + split_filename2[3] + '_' + acceleration + '.png'
            new_opti_flow_filenames.append(new_filename)
            accelerations.append(acceleration)
        else:
            print('do nothing it should never go here')
            # do nothing it should never go here

    return filenames, accelerations, new_opti_flow_filenames


def flow2rgb(flow_map, max_value):
    # flow_map_np = flow_map.detach().cpu().numpy()
    flow_map_np = flow_map.numpy()
    flow_map_np = flow_map_np.squeeze()
    flow_map_np = flow_map_np.swapaxes(0, 2)
    flow_map_np = flow_map_np.swapaxes(1, 2)
    _, h, w = flow_map_np.shape
    flow_map_np[:, (flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


# The whole transformation in Pytorch code boils down to the following function (for inference only)
def apply_transform(image):
    image = image / 255
    image = image - [0.411, 0.432, 0.45]
    return image


if __name__ == '__main__':
    None
