import os
import numpy as np


def get_filenames_labels(image_directory):
    filenames = []
    labels = []

    for file in os.listdir(image_directory):
        if(file.endswith('.jpg')):
            image_file_name = os.path.splitext(file)[0]
            split_filename = image_file_name.split('_')
            speed = split_filename[len(split_filename)-1]
            filenames.append(os.path.join(image_directory,file))
            labels.append(speed)

    print(len(filenames))
    print(len(labels))

    return filenames, labels

def flow2rgb(flow_map, max_value):
    #flow_map_np = flow_map.detach().cpu().numpy()
    flow_map_np = flow_map.numpy()
    flow_map_np = flow_map_np.squeeze()
    flow_map_np = flow_map_np.swapaxes(0,2)
    flow_map_np = flow_map_np.swapaxes(1,2)
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


if __name__ == '__main__':
    None