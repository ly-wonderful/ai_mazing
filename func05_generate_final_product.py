import cv2
import numpy as np
from moviepy.editor import *
import wave

import os 


def get_duration_wave(file_path):

    with wave.open(file_path, 'r') as audio_file:
        frame_rate = audio_file.getframerate()
        n_frames = audio_file.getnframes()
        duration = n_frames / float(frame_rate)
    
    return float(f"{duration:.2f}")

def combineAudios(bgms:list[str], voices:list[str]):

    return

def GetImageFiles():
    # Specify the folder path
    folder_path = "./AI_end/Media/Images"

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter only image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    return image_files

def GetVoiceFiles():
    # Specify the folder path
    folder_path = "./AI_end/Media/Images"

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter only image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    return image_files

def GetImageUrls():
    # Specify the folder path
    folder_path = "./AI_end/Media/Images"

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter only image files
    image_files = ["http://localhost:5000/"+file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    return image_files

def GetVoiceUrls():
    # Specify the folder path
    folder_path = "./AI_end/Media/Images"

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Filter only image files
    image_files = ["http://localhost:5000/"+file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    return image_files

def Combine( summary:list[str]):

    image_files = GetImageFiles()

    voice_files = GetVoiceFiles()

    bgm_file = "./AI_end/Media/BackgroundMusic/bgm.mp3"

    Full_Voice_Audio = 0

    VideoLength = 0

    VoicesLength = []

    for voice in voice_files:

        TimeLength = get_duration_wave(voice)

        VideoLength += TimeLength

        VoicesLength.append()

            # List of image file paths

    # Load images with durations
    clips = [ ]  # Set the duration for each image in seconds
    for index,image in enumerate(image_files):
        clips.append(ImageClip(image).set_duration(VoicesLength[index]))

    # Concatenate clips to create video
    video = concatenate_videoclips(clips)

    # Load background music
    bgm_clip = AudioFileClip("./AI_end/Media/BackgroundMusic/happy.mp3").volumex(0.5) 
    voice_clip  = AudioFileClip("clip.mp3")
        # Set the audio duration to match the video duration
    bgm_clip = bgm_clip.set_duration(video.duration)
    voice_clip = voice_clip.set_duration(video.duration)

    combined_bgm = CompositeAudioClip([bgm_file,voice_clip])

    # Combine video with subtitles and background music
    final_video = video.set_audio(combined_bgm)

    # Export the final video
    final_video.write_videofile("./AI_end/Media/Video/output_video.mp4", codec='libx264', fps=24)

    return "http://localhost:5000/AI_end/Media/Video/output_video.mp4"
