import glob
import numpy as np
import os
import re
import torch

from PIL import Image
from skimage import color


def atoi(text):
    return int(text) if text.isdigit() else text


# Natural sorting from: 
# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


class MazeDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.frames_names = MazeDataset.read_dir(root_dir)

    @staticmethod
    def read_dir(root_dir):
        """Retrieve all names of collected frames.

        We assume the following hierarchy for the collected data:

        - root_dir
            - env_0
                - ep_0
                    - 0_*.png
                    - 1_*.png
                    - ...
                - ep_1
                    - 0_*.png
                    - 1_*.png
                    ...
                - ...
            - env_1
                - ep_1
                    - 0_*.png
                    - 1_*.png
                    - ...
                - ...
            - ...

        @param root_dir: str, the root of the file path where the
            data was collected.

        @returns frames_paths: List[str], the full paths of all frames which
            would represent a valid last frame in a stack of 4.
        """
        frames_paths = []
        envs = sorted(os.listdir(root_dir), key=natural_keys)
        for env_id in envs:
            env_path = os.path.join(root_dir, str(env_id))
            eps = sorted(os.listdir(env_path), key=natural_keys)
            for ep_id in eps:
                ep_path = os.path.join(env_path, str(ep_id))
                crt_frames = os.listdir(ep_path)
                crt_frames = sorted(crt_frames, key=natural_keys)

                for frame in crt_frames[3:-1]:
                    frame_path = os.path.join(ep_path, frame)
                    frames_paths.append(frame_path)

        return frames_paths

    def get_frame_path(self, frame_dir, frame_num):
        """Get the full frame path, given a directory and a frame name.

        Because the frame names are in the format 'XXX_Y', where XXX is
        the frame index (might have more or less digits) and Y is the 
        action number, we need to be able to retrieve a frame only given
        the frame number, but not the action.

        @param frame_dir: str, representing the directory where the frame
            should be found.
        @param frame_num: str, representing the frame index we're 
            looking for.

        @returns frame_path: str, a file path inside frame_dir which
            matches the regular expression 'XXX_*', where XXX is the
            frame number. 
        """
        frame_name_re = str(frame_num) + '_*'
        frame_path_re = os.path.join(frame_dir, frame_name_re)
        frame_path = glob.glob(frame_path_re)[0]

        return frame_path

    def load_transformed_frame(self, frame_path):
        """Load a frame and transform it if neccessary.

        @param frame_path: str, the name of the frame to be read.
        @returns img: np.array with values between 0 and 1 representing
            the black and white pixel values of the frame.
        """
        img = Image.open(frame_path)
        img = np.array(img).astype(float) / 255.0

        if len(img.shape) == 3:
            img = color.rgb2gray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        """Returns a dictionary containing a dataset example.

        @param index: int, the index of the dataset example to be returned.
            The index will be used for retrieving the name of the last 
            image in the stack of four images.
        @returns data_dict: dictionary containing the following elements:
            'curr_state': np.array of shape [4, 640, 640] (if a resize 
                transform was not applied, otherwise it might vary).
            'action': int, the action that the agent took on the last frame
                from the current stack of frames.
            'next_frame': np.array of shape [640, 640] (again, if a resize
                transform was not applied)
        """
        data_dict = {
            'curr_state' : [],
            'action' : 0,
            'next_frame' : None
        }

        # Get the name of the last frame in the stack and the full
        # path of the directory where it's found.
        last_frame_path = self.frames_names[index]
        last_dir_separator = last_frame_path.rfind('/')
        last_frame_dir = last_frame_path[:last_dir_separator]
        
        # Extract all numbers from the full frame path.
        nums = re.findall(r'-?\d+\.?\d*', last_frame_path)
        _, _, last_frame_num, action_num = nums
        last_frame_num = int(last_frame_num)
        action_num = int(action_num[:-1])

        # Get the names of all names in the stack (so 3 frames in the past
        # plus the last_frame_path) and load them.
        for frame_num in range(last_frame_num - 3, last_frame_num + 1):
            frame_path = self.get_frame_path(last_frame_dir, frame_num)
            frame = self.load_transformed_frame(frame_path)
            data_dict['curr_state'].append(frame)

        # Load the next frame.
        next_frame_path = self.get_frame_path(last_frame_dir, last_frame_num + 1)
        next_frame = self.load_transformed_frame(next_frame_path)
        
        data_dict['next_frame'] = next_frame
        data_dict['action'] = action_num
        data_dict['curr_state'] = np.array(data_dict['curr_state'])

        return data_dict

    def __len__(self):
        return len(self.frames_names)


"""
# Example usage.

dataset = MazeDataset(root_dir='maze_data_cp')
train_dataloader = torch.utils.data.DataLoader(
    dataset=dataset, 
    batch_size=2, 
    shuffle=True, 
    num_workers=0, 
    drop_last=False,
)

for batch in train_dataloader:
    print(batch['curr_state'].shape)
    print(batch['action'].shape)
    print(batch['next_frame'].shape)
    break
"""