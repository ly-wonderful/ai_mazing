#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modified
@author: Gerry Huang
"""
"""
Created on Sun Jun 30 11:58:55 2024

@author: luyang
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 10:50:04 2024

@author: Mingda Wang
"""

# 安装必要的库
# =============================================================================
# !pip install openai==0.28.0
# !pip install openai
# 
# =============================================================================
# 导入库
import openai
from IPython.display import display
import requests
from PIL import Image as PILImage
from io import BytesIO
import os
import shutil

# Ensure the target directory exists
TARGET_DIR = "./AI_end/Media/Images"
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

def translate_to_prompt(paragraph):
    # place holder
    # prompt = f"Generate an illustration featuring Daddy Pig and Peppa Pig, in the style of the Peppa Pig cartoon and based on the following text:\n\n{paragraph}"
    prompt = (
        "Create an illustration featuring Daddy Pig and Peppa Pig from the Peppa Pig cartoon. "
        "Ensure that Daddy Pig and Peppa Pig look exactly as they do in the show, with consistent colors, shapes, and proportions. "
        "The style should match the original Peppa Pig cartoon, with bright colors, simple shapes, and a playful, child-friendly atmosphere. "
        "Based on the following scenario: {paragraph}"
    ) 
    return prompt.format(paragraph=paragraph)

# 定义函数来调用 DALL-E API 并显示生成的图像
# def generate_1_image_dalle(prompt):
#     response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
#     image_url = response['data'][0]['url']
#     image_response = requests.get(image_url) 
#     img = PILImage.open(BytesIO(image_response.content)).convert('RGBA') 
#     return img


def generate_1_image_dalle_new(prompt, api_key_file):
    from openai import OpenAI
    f = open(api_key_file, "r")
    api_key = f.read()
    
    client = OpenAI(api_key = api_key)
    
    response = client.images.generate(
                                      model="dall-e-3",
                                      prompt=prompt,
                                      size="1024x1024",
                                      quality="standard",
                                      n=1,
                                    )
    image_url = response.data[0].url
    image_response = requests.get(image_url) 
    img = PILImage.open(BytesIO(image_response.content)).convert('RGBA') 
    return img


def generage_image_dalle(summary, api_key_file):
    # Clear the target directory before saving new images
    clear_folder(TARGET_DIR)
    
    images = {}
    for i in range(1, len(summary) + 1):
        paragraph = summary[i]
        prompt = translate_to_prompt(paragraph)
        images[i] = generate_1_image_dalle_new(prompt, api_key_file)

        filename = os.path.join(TARGET_DIR, f"generated_image_{i}.png") 
        images[i].save(filename)
    return images
    

