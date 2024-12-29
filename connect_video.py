import os
import re
from moviepy.editor import VideoFileClip, concatenate_videoclips
from tqdm import tqdm

def sort_key(filename):
    """
    提取文件名中的 part_{} 数值和 Copy 信息，用于排序。
    """
    # 提取 part_{} 中的数值
    part_match = re.search(r"part_(\d+)", filename)
    part_num = int(part_match.group(1)) if part_match else 0

    # 提取 Copy 信息（如果有）
    copy_match = re.search(r"-Copy(\d+)", filename)
    copy_num = int(copy_match.group(1)) if copy_match else 0

    return (part_num, copy_num)

def process_videos(input_folder, output_folder):
    """
    处理输入文件夹中的视频，按照 _anaglyph.mp4 和 _sbs.mp4 分成两部分，排序并连接。
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4'))]
    if not video_files:
        print(f"No video files found in {input_folder}.")
        return

    # 分成 _anaglyph.mp4 和 _sbs.mp4 两部分
    anaglyph_files = [f for f in video_files if "_anaglyph.mp4" in f]
    sbs_files = [f for f in video_files if "_sbs.mp4" in f]

    # 对每部分进行排序
    anaglyph_files.sort(key=sort_key)
    sbs_files.sort(key=sort_key)

    # 连接 _anaglyph.mp4 视频
    if anaglyph_files:
        anaglyph_clips = []
        for file in tqdm(anaglyph_files, desc="Processing _anaglyph.mp4"):
            clip = VideoFileClip(os.path.join(input_folder, file))
            anaglyph_clips.append(clip)
        final_anaglyph = concatenate_videoclips(anaglyph_clips)
        final_anaglyph_path = os.path.join(output_folder, "final_anaglyph.mp4")
        final_anaglyph.write_videofile(final_anaglyph_path, codec="libx264")
        print(f"Saved final _anaglyph video to {final_anaglyph_path}")

    # 连接 _sbs.mp4 视频
    if sbs_files:
        sbs_clips = []
        for file in tqdm(sbs_files, desc="Processing _sbs.mp4"):
            clip = VideoFileClip(os.path.join(input_folder, file))
            sbs_clips.append(clip)
        final_sbs = concatenate_videoclips(sbs_clips)
        final_sbs_path = os.path.join(output_folder, "final_sbs.mp4")
        final_sbs.write_videofile(final_sbs_path, codec="libx264")
        print(f"Saved final _sbs video to {final_sbs_path}")

def main():
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Process videos in a folder, sort and concatenate them.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing videos.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder.")
    args = parser.parse_args()

    # 处理视频
    process_videos(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
