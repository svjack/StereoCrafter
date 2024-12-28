import os
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def split_video_by_frames(input_video_path, frames_per_segment, output_path, skip_short_segments=False):
    """
    将视频按照指定的帧数量进行分割，并保存到输出路径中。
    :param input_video_path: 输入视频文件路径
    :param frames_per_segment: 每个分割视频的帧数量
    :param output_path: 输出文件夹路径
    :param skip_short_segments: 是否跳过帧数不足的片段
    """
    # 确保输出文件夹存在
    os.makedirs(output_path, exist_ok=True)

    # 加载视频
    clip = VideoFileClip(input_video_path)
    total_frames = int(clip.fps * clip.duration)
    segment_count = 0

    # 使用 tqdm 显示进度
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(input_video_path)}") as pbar:
        for start_frame in range(0, total_frames, frames_per_segment):
            end_frame = min(start_frame + frames_per_segment, total_frames)
            start_time = start_frame / clip.fps
            end_time = end_frame / clip.fps

            # 如果跳过短片段且当前片段帧数不足，则跳过
            if skip_short_segments and (end_frame - start_frame) < frames_per_segment:
                pbar.update(frames_per_segment)
                continue

            # 截取视频片段
            segment = clip.subclip(start_time, end_time)

            # 保存分割后的视频
            segment_name = f"{os.path.splitext(os.path.basename(input_video_path))[0]}_part_{segment_count + 1}.mp4"
            segment_path = os.path.join(output_path, segment_name)
            segment.write_videofile(segment_path, codec="libx264")

            segment_count += 1
            pbar.update(frames_per_segment)

def process_videos_in_folder(input_folder, frames_per_segment, output_folder, skip_short_segments=False):
    """
    处理输入文件夹中的所有视频，按照指定的帧数量进行分割。
    :param input_folder: 输入文件夹路径
    :param frames_per_segment: 每个分割视频的帧数量
    :param output_folder: 输出文件夹路径
    :param skip_short_segments: 是否跳过帧数不足的片段
    """
    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {input_folder}.")
        return

    # 处理每个视频
    for video_file in tqdm(video_files, desc="Processing videos"):
        input_video_path = os.path.join(input_folder, video_file)
        split_video_by_frames(input_video_path, frames_per_segment, output_folder, skip_short_segments)

def main():
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Split videos into segments by frame count.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or folder containing videos.")
    parser.add_argument("--frames_per_segment", type=int, required=True, help="Number of frames per segment.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output folder.")
    parser.add_argument("--skip_short_segments", action="store_true", help="Skip segments with fewer frames than frames_per_segment.")
    args = parser.parse_args()

    # 处理输入路径
    if os.path.isfile(args.input_path):
        # 处理单个视频
        split_video_by_frames(args.input_path, args.frames_per_segment, args.output_path, args.skip_short_segments)
    else:
        # 处理文件夹中的所有视频
        process_videos_in_folder(args.input_path, args.frames_per_segment, args.output_path, args.skip_short_segments)

if __name__ == "__main__":
    main()
