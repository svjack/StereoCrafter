import argparse
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

def extract_audio_from_video(video_path, audio_output_path):
    """Extract audio from a video file and save it as an audio file."""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output_path)

def adjust_audio_speed(audio_path, video_path, output_audio_path):
    """Adjust the speed of the audio to match the length of the video."""
    audio = AudioSegment.from_file(audio_path)
    video = VideoFileClip(video_path)

    video_duration = video.duration
    audio_duration = len(audio) / 1000.0  # pydub works in milliseconds
    speed_change = audio_duration / video_duration

    changed_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed_change)
    }).set_frame_rate(audio.frame_rate)

    changed_audio.export(output_audio_path, format="mp3")

def set_audio_as_background(video_path, audio_path, output_video_path):
    """Set the modified audio as the background audio for the video."""
    video = VideoFileClip(video_path)
    modified_audio = AudioFileClip(audio_path)
    final_video = video.set_audio(modified_audio)
    final_video.write_videofile(output_video_path, codec="libx264")

def main(input_video_path, target_video_path, output_video_path):
    # Step 1: Extract audio from the input video
    temp_audio_path = "temp_audio.mp3"
    extract_audio_from_video(input_video_path, temp_audio_path)

    # Step 2: Adjust the audio speed to match the target video length
    adjusted_audio_path = "adjusted_audio.mp3"
    adjust_audio_speed(temp_audio_path, target_video_path, adjusted_audio_path)

    # Step 3: Set the adjusted audio as the background for the target video
    set_audio_as_background(target_video_path, adjusted_audio_path, output_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adjust audio from one video to match the length of another video and set it as the background audio.")
    parser.add_argument("input_video", help="Path to the input video from which audio will be extracted.")
    parser.add_argument("target_video", help="Path to the target video whose length the audio will match.")
    parser.add_argument("output_video", help="Path to the output video with the adjusted audio.")
    args = parser.parse_args()

    main(args.input_video, args.target_video, args.output_video)