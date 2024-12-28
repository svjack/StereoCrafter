import os
import argparse
import numpy as np
from moviepy.editor import ImageSequenceClip, VideoFileClip
from tqdm import tqdm

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
    print(f"Loaded {npz_path} with shape: {video_grid.shape}")

    # 检查是否为黑白视频（单通道）
    if len(video_grid.shape) == 3:  # 形状为 (T, H, W)
        # 将黑白视频转换为三通道
        video_grid = np.stack([video_grid] * 3, axis=-1)  # 形状变为 (T, H, W, 3)
        print(f"Converted to 3-channel video with shape: {video_grid.shape}")

    # 使用 moviepy 保存视频
    with tqdm(total=1, desc="Saving video") as pbar:
        clip = ImageSequenceClip(list(video_grid), fps=fps)
        clip.write_videofile(output_video_path, codec="libx264", ffmpeg_params=["-crf", "16"])
        pbar.update(1)

def resize_video(input_video_path, output_video_path, scale_factor=0.5):
    """
    缩放视频的长宽
    :param input_video_path: 输入视频文件路径
    :param output_video_path: 输出视频文件路径
    :param scale_factor: 缩放因子，例如 0.5 表示长宽变为原来的一半，2 表示长宽变为原来的两倍
    """
    # 加载视频
    with tqdm(total=1, desc="Loading video") as pbar:
        clip = VideoFileClip(input_video_path)
        pbar.update(1)

    # 计算新的分辨率
    new_width = int(clip.size[0] * scale_factor)
    new_height = int(clip.size[1] * scale_factor)

    # 缩放视频
    with tqdm(total=1, desc="Resizing video") as pbar:
        resized_clip = clip.resize((new_width, new_height))
        pbar.update(1)

    # 保存缩放后的视频
    with tqdm(total=1, desc="Saving resized video") as pbar:
        resized_clip.write_videofile(output_video_path, codec="libx264")
        pbar.update(1)

def process_npz_folder(input_folder, output_folder, fps=30, key="video_grid", scale_factor=0.5, suffix="_video_grid.npz"):
    """
    处理输入文件夹中的所有 npz 文件，将其转换为视频并调整分辨率
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param fps: 视频帧率
    :param key: npz 文件中保存视频数据的键名
    :param scale_factor: 视频分辨率缩放因子
    :param suffix: npz 文件的尾部匹配字符串
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有符合条件的 npz 文件
    npz_files = [f for f in os.listdir(input_folder) if f.endswith(suffix)]
    if not npz_files:
        print(f"No files found with suffix '{suffix}' in {input_folder}.")
        return

    # 使用 tqdm 遍历并处理每个 npz 文件
    for npz_file in tqdm(npz_files, desc="Processing npz files"):
        npz_path = os.path.join(input_folder, npz_file)
        video_name = os.path.splitext(npz_file)[0] + ".mp4"
        temp_video_path = os.path.join(output_folder, video_name)
        final_video_path = os.path.join(output_folder, f"resized_{video_name}")

        # 将 npz 文件转换为视频
        npz_to_video(npz_path, temp_video_path, fps, key)

        # 调整视频分辨率
        resize_video(temp_video_path, final_video_path, scale_factor)

        # 删除临时视频文件
        os.remove(temp_video_path)
        print(f"Saved resized video to {final_video_path}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Convert npz files to videos and resize them.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing npz files.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder for saving videos.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate of the output videos.")
    parser.add_argument("--key", type=str, default="video_grid", help="Key name for video data in npz files.")
    parser.add_argument("--scale_factor", type=float, default=0.5, help="Scale factor for resizing videos (e.g., 0.5 for half size).")
    parser.add_argument("--suffix", type=str, default="_video_grid.npz", help="Suffix to filter npz files (e.g., '_video_grid.npz').")
    args = parser.parse_args()

    # 处理 npz 文件夹
    process_npz_folder(args.input_folder, args.output_folder, args.fps, args.key, args.scale_factor, args.suffix)

if __name__ == "__main__":
    main()
