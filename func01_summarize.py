#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:59:24 2024

@author: Zhaoyan Zhi, Liang Han
"""

from openai import OpenAI
import requests

import mimetypes
import fitz
from docx import Document
import re

import openai
from pdfminer.high_level import extract_text
import re
import os
from openai import OpenAI

from odf.opendocument import load
from odf.text import H, P
from odf.element import Element


# Define methods that read an input file in different format
def read_pdf(file_path):
    """
    Reads the content of a PDF file.
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text


def read_docx(file_path):
    """
    Reads the content of a DOCX file.
    """
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def read_txt(file_path):
    """
    Reads the content of a TXT file.
    """
    file_input = open(file_path, 'r', encoding='UTF-8')
    text = file_input.read()
    return text


def read_document(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type == 'application/pdf':
        return read_pdf(file_path)
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return read_docx(file_path)
    elif mime_type == 'text/plain':
        return read_txt(file_path)
    else:
        return "File type not supported."


def engineer_prompt_summary():
    #prompt engineering part
    prompt_preset = 'give me 10 sentences to summarize the story, using simple words so that reader with elementory school level vocabulary can understand'
    
    return prompt_preset


def divide_paragraphs(text):
    # divide the text to multiple paragraphs and save as a Python dictionary
    # Each page in the final output has one paragraph.
    # 
    # MVP version is to divide by period.
    sliced_text = {}
    for i, p in enumerate(text.split('.'), start=1):
        if p != '':
            sliced_text[i] = p + '.'
        else:
            sliced_text[i] = 'The End'
        
    return sliced_text


# Define a method that summarize a long text
def summarize_text(file_path, api_key = "chatgpt_api_key.txt", max_tokens=500, model="gpt-3.5-turbo"):
    
    text = read_document(file_path)
    
    api_file = open(api_key, "r")
    api_key_str = api_file.read()
    client = OpenAI(api_key = api_key_str)
    
    prompt_preset = engineer_prompt_summary()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"{prompt_preset}:\n\n{text}\n\nKids Story:",
            }
        ],
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7,
    )
    summary = divide_paragraphs(response.choices[0].message.content)
    return summary


"""# turn text into conversation of peppa"""
def paper_text_to_conversation(text,api_key_file):

    api_file = open(api_key_file, "r")
    api_key_str = api_file.read()
    
    """# 清理文本中的噪声字符"""
    def clean_text(text):
        # 移除所有的换行符，并将其替换为单个空格
        text = re.sub(r'\n+', ' ', text)
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
        # 移除 <EOS> 和 <pad> 标签
        text = re.sub(r'<EOS>|<pad>', ' ', text)
        return text
    
    """#清楚reference部分"""
    def remove_references(text):
        # 匹配“References”或类似的标题，并且它应该接近文本末尾
        reference_pattern = re.compile(r'\b(References|Bibliography|Works Cited)\b', re.IGNORECASE)
        # 搜索标题在文本中的位置
        match = reference_pattern.search(text)

        if match:
            # 提取引用部分开始的索引
            start_index = match.start()
            # 检查引用部分是否包含多个典型的引用条目
            # 一个简单的匹配模式来识别引用条目，例如: [1], [2], ... 或者 1., 2., ...
            entries_pattern = re.compile(r'\[\d+\]|\d+\.\s', re.IGNORECASE)
            if entries_pattern.search(text, start_index):
                # 截取从开头到引用部分开始前的文本
                text_without_references = text[:start_index]
                return text_without_references.strip()
        # 如果没有找到引用部分或没有发现引用条目，返回原文本
        return text
    
    """#remove permission statement"""
    def remove_permission_statement(text, statement):
        """
        Remove the specific permission statement from the text.

        :param text: The original text containing the permission statement.
        :param statement: The permission statement to be removed.
        :return: The text with the permission statement removed.
        """
        # Remove the specific statement from the text
        modified_text = text.replace(statement, '')
        return modified_text

    """split text """
    def split_text_with_token_count(text, max_tokens=2048, overlap_tokens=200):
        words = text.split()
        chunks = []
        current_chunk = []
        current_chunk_token_count = 0
        overlap_buffer = []

        for word in words:
            # Check the token count for the word
            token_count = 1.5
            if current_chunk_token_count + token_count <= max_tokens:
                current_chunk.append(word)
                current_chunk_token_count += token_count
            else:
                # Add the current chunk to the list of chunks
                chunks.append(' '.join(current_chunk))
                # Prepare the overlap buffer
                overlap_buffer = current_chunk[-overlap_tokens:] if len(current_chunk) >= overlap_tokens else current_chunk[:]
                # Start a new chunk
                current_chunk = overlap_buffer + [word]
                current_chunk_token_count = len(current_chunk)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
    
    """Genearte summary for each paragraph and combine"""
    def generate_comprehensive_summary(paper_text_paragraphs,api_key_str):

        def summarize_text(text,api_key_str):
            client = OpenAI(
            api_key = api_key_str,
        )

            completion = client.completions.create(
              model = "gpt-3.5-turbo-instruct"
              #,prompt =        f"Please summarize the following text in exactly {num_paragraphs} sentences:\n\n{text} "
              ,prompt = f"Your task is to summarize the following text into a shorter version with words more than 500 words:"
                f"\n\n{text}\n\n"
            #f"Ensure the output should be longer than 800 words"
                ,max_tokens = 1000
              ,temperature = 0.5 #0.7 # high temp 1 or above, low tep 0.2-0.5
              ,top_p=1.0
            )
            return completion.choices[0].text   


        comprehensive_summary = ""
        for i in range(len(paper_text_paragraphs)):

            solo_summary = summarize_text(paper_text_paragraphs[i],api_key_str)
            print(solo_summary)
            comprehensive_summary = comprehensive_summary + "\n\n" + solo_summary

        return comprehensive_summary
    
    """turn summary into conversation"""
    def conversation_summary(comprehensive_summary,api_key):
        import os
        from openai import OpenAI

        # 设置你的API密钥
        api_key_str = api_key

        client = OpenAI(
        api_key = api_key_str)

        input_text=comprehensive_summary

        completion = client.completions.create(
          model = "gpt-3.5-turbo-instruct"
          #,prompt =        f"Please summarize the following text in exactly {num_paragraphs} sentences:\n\n{text} "
          ,prompt = 
        f"Your task is to rewrite the following text into a 500-word summary that is easy to understand for a five-year-old child:\n\n"
        f"{input_text}\n\n"
        f"Ensure this summary is presented in the form of a conversation between Peppa Pig and Daddy Pig, keeping it very cute and engaging. "
        f"Use simple language and short sentences. "
        f"Do not inlcude content related to copy notice or citation guide in the summary, and do not include content like 'company gave permission for people to use the paper result' "
        f"Ensure that whenever a complex concept or terminology appears, have Peppa ask a question about it, and Daddy Pig should explain it in an interesting and age-appropriate and easy-to-understand way. "
        f"Do not mention more than 5 complicated concepts or teminologies so that the output can be easily undertood by kids."
        f"Remember to maintain the playful and friendly tone typical of Peppa Pig episodes."
            ,max_tokens = 1000
          ,temperature = 0.5 #0.7 # high temp 1 or above, low tep 0.2-0.5
          ,top_p=1.0
        )

        return completion.choices[0].text.strip()
    
    """turn conversation output into dictionary"""
    def turn_conversation_into_dict(text):
        text_list=text.split('\n\n')
        text_dict=dict()

        for i in range(len(text_list)):
            text_dict[i+1]=text_list[i]

        return text_dict

    
    """Run functions"""
    paper_text=clean_text(text)
    paper_text=remove_references(paper_text)
    
    permission_statement = "Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works."
    tpaper_text=remove_permission_statement(paper_text, permission_statement)
    
    paper_text_paragraphs=split_text_with_token_count(paper_text, max_tokens=2048, overlap_tokens=200)
    
    comprehensive_summary=generate_comprehensive_summary(paper_text_paragraphs,api_key_str)
    
    conversation=conversation_summary(comprehensive_summary,api_key_str)
    
    conversation_in_dict=turn_conversation_into_dict(conversation)
    
    return conversation_in_dict