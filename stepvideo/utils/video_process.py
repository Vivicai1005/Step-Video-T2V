import numpy as np
import datetime
import torch
import os
import imageio
import numpy as np
from einops import rearrange
from PIL import Image
from tqdm import tqdm


class VideoProcessor:
    def __init__(self, save_path: str='./results', name_suffix: str=''):
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.name_suffix = name_suffix
    
    def crop2standard540p(self, vid_array):
        _, height, width, _ = vid_array.shape
        height_center = height//2
        width_center = width//2
        if width_center>height_center:  ## horizon mode
            return vid_array[:, height_center-270:height_center+270, width_center-480:width_center+480]
        elif width_center<height_center: ## portrait mode
            return vid_array[:, height_center-480:height_center+480, width_center-270:width_center+270]
        else:
            return vid_array

    def save_imageio_video(self, video_array: np.array, output_filename: str, fps=25, codec='libx264'):
        
        ffmpeg_params = [
            "-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1",  # denoise
        ]
   
        with imageio.get_writer(output_filename, fps=fps, codec=codec, ffmpeg_params=ffmpeg_params) as vid_writer:
            for img_array in video_array:
                vid_writer.append_data(img_array)

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def save_video(self, frames, save_path, fps=25, quality=9):
        ffmpeg_params = [
            "-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1",  # denoise
        ]
        writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            writer.append_data(frame)
        writer.close()

    def postprocess_video(self, video_tensor, output_file_name='', output_type="mp4", crop2standard540p=False):
        if len(self.name_suffix) == 0:
            video_path = os.path.join(self.save_path, f"{output_file_name}-{str(datetime.datetime.now())}.{output_type}")
        else:
            video_path = os.path.join(self.save_path, f"{output_file_name}-{self.name_suffix}.{output_type}")
        
        video_array = self.tensor2video(video_tensor)
        
        if crop2standard540p:
            video_array = self.crop2standard540p(video_array)

        self.save_video(frames=video_array, save_path=video_path, fps=25, quality=5)
        print(f"Saved the generated video in {video_path}")