import os
import re
import uuid
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

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

def process_videos(input_folder, output_folder, group_by_type=None, connected_video=None):
    """
    处理输入文件夹中的视频，记录路径并重命名。
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param group_by_type: 分组类型（"_anaglyph.mp4" 或 "_sbs.mp4"）
    :param connected_video: 额外的视频文件路径，会排在第一位
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4'))]
    if not video_files and not connected_video:
        print(f"No video files found in {input_folder}.")
        return

    # 创建 videos 文件夹
    videos_dir = os.path.join(output_folder, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # 准备 metadata.csv 数据
    metadata_data = []

    # 处理额外的视频文件
    if connected_video:
        if not os.path.exists(connected_video):
            print(f"警告: 额外的视频文件 {connected_video} 不存在，已跳过。")
        else:
            # 生成新的文件名
            new_filename = f"{uuid.uuid4()}.mp4"
            # 目标视频路径
            dst_video_path = os.path.join(videos_dir, new_filename)
            # 复制视频文件（使用 shutil.copy2 替换 os.rename）
            shutil.copy2(connected_video, dst_video_path)
            # 添加到 metadata.csv 数据
            metadata_data.append({
                "type": "connected_video",  # type 排在第一位
                "sort_key": (0, 0),        # sort_key 排在第二位
                "file_name": f"videos/{new_filename}"  # file_name 排在最后
            })

    # 处理输入文件夹中的视频文件
    if group_by_type in ["_anaglyph.mp4", "_sbs.mp4"]:
        # 过滤出指定类型的视频文件
        filtered_files = [f for f in video_files if group_by_type in f]
        # 对文件进行排序
        filtered_files.sort(key=sort_key)

        # 处理指定类型的视频文件
        for file in tqdm(filtered_files, desc=f"Processing {group_by_type}"):
            # 生成新的文件名
            new_filename = f"{uuid.uuid4()}.mp4"
            # 源视频路径
            src_video_path = os.path.join(input_folder, file)
            # 目标视频路径
            dst_video_path = os.path.join(videos_dir, new_filename)
            # 复制视频文件（使用 shutil.copy2 替换 os.rename）
            shutil.copy2(src_video_path, dst_video_path)
            # 添加到 metadata.csv 数据
            metadata_data.append({
                "type": group_by_type,      # type 排在第一位
                "sort_key": sort_key(file), # sort_key 排在第二位
                "file_name": f"videos/{new_filename}"  # file_name 排在最后
            })
    else:
        print("警告: 未指定有效的分组类型，已跳过视频处理。")

    # 保存 metadata.csv
    metadata = pd.DataFrame(metadata_data)
    metadata.to_csv(os.path.join(output_folder, "metadata.csv"), index=False)

    print(f"文件准备完成，路径: {output_folder}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Process videos in a folder, sort and rename them.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing videos.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--group_by_type", type=str, choices=["_anaglyph.mp4", "_sbs.mp4"], help="Group videos by _anaglyph.mp4 or _sbs.mp4.")
    parser.add_argument("--connected_video", type=str, help="Path to a connected video file to be added at the top.")
    args = parser.parse_args()

    # 处理视频
    process_videos(args.input_folder, args.output_folder, args.group_by_type, args.connected_video)

if __name__ == "__main__":
    main()