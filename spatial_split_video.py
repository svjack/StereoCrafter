import os
import argparse
from moviepy.editor import VideoFileClip

def spatial_split_video(input_video_path, split_direction):
    """
    将视频在空间维度上（长宽）分割成两部分，并保存到输出路径中。
    :param input_video_path: 输入视频文件路径
    :param split_direction: 分割方向，'horizontal' 或 'vertical'
    """
    # 加载视频
    clip = VideoFileClip(input_video_path)
    width, height = clip.size

    # 根据分割方向进行分割
    if split_direction == "horizontal":
        # 水平分割（上下两部分）
        part1_clip = clip.crop(y1=0, y2=height // 2)
        part2_clip = clip.crop(y1=height // 2, y2=height)
        suffix1 = "_top.mp4"
        suffix2 = "_bottom.mp4"
    elif split_direction == "vertical":
        # 竖直分割（左右两部分）
        part1_clip = clip.crop(x1=0, x2=width // 2)
        part2_clip = clip.crop(x1=width // 2, x2=width)
        suffix1 = "_left.mp4"
        suffix2 = "_right.mp4"
    else:
        raise ValueError("Invalid split direction. Use 'horizontal' or 'vertical'.")

    # 生成输出文件路径
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_dir = os.path.dirname(input_video_path)
    output_path1 = os.path.join(output_dir, base_name + suffix1)
    output_path2 = os.path.join(output_dir, base_name + suffix2)

    # 保存分割后的视频
    part1_clip.write_videofile(output_path1, codec="libx264")
    part2_clip.write_videofile(output_path2, codec="libx264")
    print(f"Saved split videos to {output_path1} and {output_path2}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Split a video spatially (by width or height) into two parts.")
    parser.add_argument("--input_video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--split_direction", type=str, required=True, choices=["horizontal", "vertical"], help="Direction to split the video: 'horizontal' or 'vertical'.")
    args = parser.parse_args()

    # 处理视频
    spatial_split_video(args.input_video_path, args.split_direction)

if __name__ == "__main__":
    main()