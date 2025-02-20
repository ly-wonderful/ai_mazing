#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===================================================================
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# modified
# @author: Gerry Huang
# """
# """
# Created on Sun Jun 30 12:01:29 2024
# 
# @author: Rin Hayashi
# """
#
#
# Installation of libarary============================================
# !pip install simpleaudio  
# !pip install vaderSentiment
#!pip install pydub
#=====================================================================
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

from pydub import AudioSegment
import simpleaudio as sa
import os
import shutil

# Ensure the target directory exists
TARGET_DIR = "./AI_end/Media/BackgroundMusic"
os.makedirs(TARGET_DIR, exist_ok=True)

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
        
# ===Function to analyze summary and generate sentiment
#
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment
#
# === Function needed when play the mp3 music file
# def play_mp3(file_path):

#      audio = AudioSegment.from_file(file_path, format="mp3")

# # Play the audio file
#      play_obj = sa.play_buffer(audio.raw_data, 
#                           num_channels=audio.channels,
#                           bytes_per_sample=audio.sample_width,
#                           sample_rate=audio.frame_rate)
#
# === Function to genetate BGM based on music, the prepared mp3 files are for 20 seconds
#    
def generate_bgm(summary_text, api_key_file):
    # Clear the target directory before saving new music
    clear_folder(TARGET_DIR)

    from openai import OpenAI
    f = open(api_key_file, "r")
    api_key = f.read()
    
    client = OpenAI(api_key = api_key)

    # Combine all strings in the text dictionary into a single string
    combined_text = " ".join(summary_text.values())
    
    # Use OpenAI GPT-3.5 API to summarize the combined text
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Summarize the following text in a way that is suitable for sentiment analysis:\n\n{combined_text}"}
    ],
    max_tokens=100
    )

    summary_text = response.choices[0].message.content.strip()

    sentiment = get_sentiment(summary_text)
    # Determine which MP3 file to play based on sentiment
    if sentiment['compound'] >= 0.05:
        bgm = 'happy.mp3'  # Replace with the path to your positive sentiment MP3 file
    elif sentiment['compound'] <= -0.05:
        bgm = 'sad.mp3'  # Replace with the path to your negative sentiment MP3 file
    else:
        bgm = 'neutral.mp3'  # Replace with the path to your neutral sentiment MP3 file

     # Ensure the target directory exists and copy the BGM file to it
    bgm_source_path = os.path.join('./AI_end/generate-audio/generated music', bgm)  # Adjust this path
    bgm_target_path = os.path.join(TARGET_DIR, bgm)

    shutil.copy(bgm_source_path, bgm_target_path)

    return bgm
    
#if __name__ == '__main__':
#      summary_text = input("Enter text to analyze sentiment: ")
#
##    print(f"Playing {bgm} for sentiment: {sentiment['compound']}")
#      bgm=generate_bgm(summary_text)
#      play_mp3(bgm)
    


# In[ ]:





# In[ ]:




