# =============================================================================
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sun Jun 30 12:01:29 2024
# 
# @author: Qiong Li, Xiang
# """

import os
import re
import json
import boto3
import wave
from google.cloud import language_v1
from botocore.exceptions import BotoCoreError, ClientError
import gender_guesser.detector as gender

class TextToSpeech:
    def __init__(self, google_credentials_path, polly_client):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
        self.language_client = language_v1.LanguageServiceClient()
        self.polly_client = polly_client
        self.gender_detector = gender.Detector()

    def analyze_sentiments(self, text):
        sentences = []
        matches = re.findall(r"([^:]+): \"([^\"]+)\"", text)
        for speaker, dialogue in matches:
            sentences.append((speaker.strip(), dialogue))
        return sentences

    def choose_voice(self, speaker):
        first_name = speaker.split()[0].strip()
        gender_guess = self.gender_detector.get_gender(first_name)
        if first_name in ['Mummy', 'Mom']:
            return 'Kendra'
        elif gender_guess in ['female', 'mostly_female'] and first_name in ['Peppa']:
            return 'Salli'
        elif first_name in ['Daddy', 'Dad']:
            return 'Matthew'
        elif gender_guess in ['male', 'mostly_male']:
            return 'Justin'
        return 'Joanna'

    def create_ssml(self, sentences):
        ssml_text = "<speak>"
        for speaker, dialogue in sentences:
            ssml_text += f"<prosody rate='slow' pitch='medium'>{dialogue}</prosody>"
            ssml_text += "<break time='800ms'/>"
        ssml_text += "</speak>"
        return ssml_text

    def synthesize_speech(self, paragraphs, output_dir='output'):
        audio_files = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, paragraph in enumerate(paragraphs):
            speaker, dialogues = zip(*paragraph)
            ssml_text = self.create_ssml(paragraph)
            voice_id = self.choose_voice(speaker[0])
            try:
                response = self.polly_client.synthesize_speech(
                    Text=ssml_text,
                    OutputFormat='pcm',
                    VoiceId=voice_id,
                    TextType='ssml'
                )

                # Save the PCM data to a WAV file
                wav_filename = os.path.join(output_dir, f'paragraph_{i + 1}.wav')
                with wave.open(wav_filename, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono audio
                    wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                    wav_file.setframerate(16000)  # 16000 samples per second
                    wav_file.writeframes(response['AudioStream'].read())
                audio_files.append(wav_filename)
            except (BotoCoreError, ClientError) as error:
                print(f"An error occurred: {error}")

        return audio_files

def main():
    google_credentials_path = r'C:\Users\qiong\Documents\Chase_AI_team\Credenetials\sentiment-auth.json'
    original_path = r'C:\Users\qiong\Documents\Chase_AI_team\Texts\sample_text.txt'
    
    with open(original_path, 'r') as file:
        text = file.read()
        
    with open(r'C:\Users\qiong\Documents\Chase_AI_team\Credenetials\AWS_auth.json') as f:
        credentials = json.load(f)

    polly_client = boto3.client(
        'polly',
        aws_access_key_id=credentials['accessKeyId'],
        aws_secret_access_key=credentials['secretAccessKey'],
        region_name='us-east-2'
    )

    tts = TextToSpeech(google_credentials_path, polly_client)

    sentences = tts.analyze_sentiments(text)

    paragraphs = [sentences[i:i + 1] for i in range(0, len(sentences), 1)]

    audio_files = tts.synthesize_speech(paragraphs)

    print(f"Generated audio files: {audio_files}")

if __name__ == "__main__":
    main()
# 
# 
# =============================================================================
