
<div align="center">
<h2>StereoCrafter: Diffusion-based Generation of Long and High-fidelity Stereoscopic 3D from Monocular Videos</h2>

Sijie Zhao*&emsp;
Wenbo Hu*&emsp;
Xiaodong Cun*&emsp;
Yong Zhang&dagger;&emsp;
Xiaoyu Li&dagger;&emsp;<br>
Zhe Kong&emsp;
Xiangjun Gao&emsp;
Muyao Niu&emsp;
Ying Shan

&emsp;* equal contribution &emsp; &dagger; corresponding author 

<h3>Tencent AI Lab&emsp;&emsp;ARC Lab, Tencent PCG</h3>

<a href='https://arxiv.org/abs/2409.07447'><img src='https://img.shields.io/badge/arXiv-PDF-a92225'></a> &emsp;
<a href='https://stereocrafter.github.io/'><img src='https://img.shields.io/badge/Project_Page-Page-64fefe' alt='Project Page'></a> &emsp;
<a href='https://huggingface.co/TencentARC/StereoCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow'></a>
</div>

## 💡 Abstract

We propose a novel framework to convert any 2D videos to immersive stereoscopic 3D ones that can be viewed on different display devices, like 3D Glasses, Apple Vision Pro and 3D Display. It can be applied to various video sources, such as movies, vlogs, 3D cartoons, and AIGC videos.

![teaser](assets/teaser.jpg)

## 📣 News
- `2024/12/27` We released our inference code and model weights.
- `2024/09/11` We submitted our technical report on arXiv and released our project page.

## 🎞️ Showcases
Here we show some examples of input videos and their corresponding stereo outputs in Anaglyph 3D format.
<div align="center">
    <img src="assets/demo.gif">
</div>


## 🛠️ Installation

#### 1. Set up the environment
We run our code on Python 3.8 and Cuda 11.8.
You can use Anaconda or Docker to build this basic environment.

#### 2. Clone the repo
```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm
sudo apt-get upgrade ffmpeg

conda create -n stereoCrafter python=3.10
conda activate stereoCrafter
pip install ipykernel
python -m ipykernel install --user --name stereoCrafter --display-name "stereoCrafter"

git clone https://github.com/svjack/StereoCrafter
cd StereoCrafter
```

# When run depth_splatting_inference_npz.py

#### 3. Install the requirements
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip uninstall xformers
pip install xformers
pip install moviepy==1.0.3
pip install pillow==9.0.0
```


#### 4. Install customized 'Forward-Warp' package for forward splatting
```
cd ./dependency/Forward-Warp
chmod a+x install.sh
./install.sh
```

# When run inpainting_inference.py

```bash
pip install -r requirements.txt
```

## 📦 Model Weights

#### 1. Download the [SVD img2vid model](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for the image encoder and VAE.

```bash
# in StereoCrafter project root directory
mkdir weights
cd ./weights
git lfs install

huggingface-cli login
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 --local-dir stable-video-diffusion-img2vid-xt-1-1
#### OR
git clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1
```

#### 2. Download the [DepthCrafter model](https://huggingface.co/tencent/DepthCrafter) for the video depth estimation.
```bash
git clone https://huggingface.co/tencent/DepthCrafter
```

#### 3. Download the [StereoCrafter model](https://huggingface.co/TencentARC/StereoCrafter) for the stereo video generation.
```bash
git clone https://huggingface.co/TencentARC/StereoCrafter
```


## 🔄 Inference
- Step 0
```bash
python depth_splatting_inference_npz.py \
   --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1\
   --unet_path ./weights/DepthCrafter \
   --input_video_path ./source_video/camel.mp4 \
   --output_video_path ./outputs/camel_splatting_results.mp4
```
- Step 1
```python
import numpy as np
from moviepy.editor import ImageSequenceClip

def npz_to_video(npz_path, output_video_path, fps, key="video_grid"):
    """
    从 npz 文件中加载视频结构并保存为视频文件
    :param npz_path: npz 文件路径
    :param output_video_path: 输出视频文件路径
    :param fps: 视频帧率
    :param key: npz 文件中保存视频数据的键名
    """
    # 加载 npz 文件
    data = np.load(npz_path)
    video_grid = data[key]  # 假设视频结构保存在 'video_grid' 键中

    # 将视频结构转换为 0-255 范围的图像序列
    video_grid = (video_grid * 255).astype(np.uint8)
    print("Original shape:", video_grid.shape)

    # 检查是否为黑白视频（单通道）
    if len(video_grid.shape) == 3:  # 形状为 (T, H, W)
        # 将黑白视频转换为三通道
        video_grid = np.stack([video_grid] * 3, axis=-1)  # 形状变为 (T, H, W, 3)
        print("Converted shape:", video_grid.shape)

    # 使用 moviepy 保存视频
    clip = ImageSequenceClip(list(video_grid), fps=fps)
    clip.write_videofile(output_video_path, codec="libx264", ffmpeg_params=["-crf", "16"])

# 示例调用
npz_path = "outputs/camel_splatting_results_video_grid.npz"  # 替换为你的 npz 文件路径
output_video_path = "outputs/camel_splatting_results_video_grid.mp4"  # 替换为输出视频文件路径
fps = 30  # 替换为视频的帧率
npz_to_video(npz_path, output_video_path, fps)

#### OR

# 示例调用
npz_path = "outputs/camel_splatting_results.npz"  # 替换为你的 npz 文件路径
output_video_path = "outputs/camel_splatting_results.mp4"  # 替换为输出视频文件路径
fps = 30  # 替换为视频的帧率
npz_to_video(npz_path, output_video_path, fps, key = "depth")
```

camel_splatting_results_video_grid.mp4


https://github.com/user-attachments/assets/160f2be6-a069-4abd-9c35-045c5565d17d

camel_splatting_results.mp4


https://github.com/user-attachments/assets/85a34c1e-b2f9-40e5-9cdd-ff844f1449b8

```bash
cp ./outputs/camel_splatting_results_video_grid.mp4 camel_splatting_results_video_grid.mp4
```

```python
from moviepy.editor import VideoFileClip

def resize_video(input_video_path, output_video_path, scale_factor=0.5):
    """
    缩放视频的长宽
    :param input_video_path: 输入视频文件路径
    :param output_video_path: 输出视频文件路径
    :param scale_factor: 缩放因子，例如 0.5 表示长宽变为原来的一半，2 表示长宽变为原来的两倍
    """
    # 加载视频
    clip = VideoFileClip(input_video_path)

    # 计算新的分辨率
    new_width = int(clip.size[0] * scale_factor)
    new_height = int(clip.size[1] * scale_factor)

    # 缩放视频
    resized_clip = clip.resize((new_width, new_height))

    # 保存缩放后的视频
    resized_clip.write_videofile(output_video_path, codec="libx264")

# 示例调用
input_video_path = "camel_splatting_results_video_grid.mp4"  # 替换为输入视频路径
output_video_path = "camel_splatting_results_video_grid_half.mp4"  # 替换为输出视频路径
scale_factor = 0.5  # 缩放因子，0.5 表示长宽变为原来的一半，2 表示长宽变为原来的两倍
resize_video(input_video_path, output_video_path, scale_factor)
```

camel_splatting_results_video_grid_half.mp4



https://github.com/user-attachments/assets/1430695c-a48e-47e7-9057-7022bd6a79d7



  
- Step 2
```
python inpainting_inference.py \
    --pre_trained_path ./weights/stable-video-diffusion-img2vid-xt-1-1 \
    --unet_path ./weights/StereoCrafter \
    --input_video_path camel_splatting_results_video_grid_half.mp4 \
    --save_dir ./outputs
```
camel_video_grid_half_inpainting_results_sbs.mp4



https://github.com/user-attachments/assets/6c0308ba-270b-4feb-aa9c-9ba45f9ce542



camel_video_grid_half_inpainting_results_anaglyph.mp4






https://github.com/user-attachments/assets/4a52bb9f-0c14-4a1d-8aee-8975f62fd447




Script:

```bash
# in StereoCrafter project root directory
sh run_inference.sh
```

There are two main steps in this script for generating stereo video.

#### 1. Depth-Based Video Splatting Using the Video Depth from DepthCrafter
Execute the following command:
```bash
python depth_splatting_inference.py --pre_trained_path [PATH] --unet_path [PATH]
                                    --input_video_path [PATH] --output_video_path [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the DepthCrafter model weights (e.g., `./weights/DepthCrafter`).
- `--input_video_path`: Path to the input video (e.g., `./source_video/camel.mp4`).
- `--output_video_path`: Path to the output video (e.g., `./outputs/camel_splatting_results.mp4`).
- `--max_disp`: Parameter controlling the maximum disparity between the generated right video and the input left video. Default value is `20` pixels.

The first step generates a video grid with input video, visualized depth map, occlusion mask, and splatting right video, as shown below:

<img src="assets/camel_splatting_results.jpg" alt="camel_splatting_results" width="800"/> 

#### 2. Stereo Video Inpainting of the Splatting Video
Execute the following command:
```bash
python inpainting_inference.py --pre_trained_path [PATH] --unet_path [PATH]
                               --input_video_path [PATH] --save_dir [PATH]
```
Arguments:
- `--pre_trained_path`: Path to the SVD img2vid model weights (e.g., `./weights/stable-video-diffusion-img2vid-xt-1-1`).
- `--unet_path`: Path to the StereoCrafter model weights (e.g., `./weights/StereoCrafter`).
- `--input_video_path`: Path to the splatting video result generated by the first stage (e.g., `./outputs/camel_splatting_results.mp4`).
- `--save_dir`: Directory for the output stereo video (e.g., `./outputs`).
- `--tile_num`: The number of tiles in width and height dimensions for tiled processing, which allows for handling high resolution input without requiring more GPU memory. The default value is `1` (1 $\times$ 1 tile). For input videos with a resolution of 2K or higher, you could use more tiles to avoid running out of memory.

The stereo video inpainting generates the stereo video result in side-by-side format and anaglyph 3D format, as shown below:

<img src="assets/camel_sbs.jpg" alt="camel_sbs" width="800"/> 

<img src="assets/camel_anaglyph.jpg" alt="camel_anaglyph" width="400"/>

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects:
- [Stable Video Diffusion](https://github.com/Stability-AI/generative-models): A latent diffusion model trained to generate video clips from an image or text conditioning.
- [DepthCrafter](https://github.com/Tencent/DepthCrafter): A novel method to generate temporally consistent depth sequences from videos.


## 📚 Citation

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```
