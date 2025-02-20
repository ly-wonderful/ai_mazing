#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 11:53:46 2024

@author: luyang
edited by: Gerry Huang
"""

# import py packages
import os

# import all functions
from func01_summarize import paper_text_to_conversation
import func02_generate_image as image
import func03_generate_music as music
import func04_generate_voice as voice
import func05_generate_final_product as result

from flask import Flask, request, jsonify


# Global Input - a user-specified document
file_path = 'THE LOCAL MULTIPLIER IN THE STATE OF OHIO.docx'



app = Flask(__name__)

@app.route('/generate', methods=['POST'])

def main():

    Content = request.get_json()["Content"]

    # Module 1 - Generate summarization
    # Input: user-specified document
    # Output: 'summary' - a Python dictionary
    # Description: Each value of 'summary' is the text on each page of the final product.
    #              indexed by numbers starting from 1.

    chatgpt_api_key_file = "chatgpt_api_key.txt"
    summary=paper_text_to_conversation(Content,chatgpt_api_key_file)

    # Module 2 - Generate images
    # Input: 'summary'
    # Output: 'images' - a Python dictionary
    # Description: Each value of 'images' is the image on each page of the final product.
    #              indexed by numbers starting from 1.
    # summary = {1: "Peppa Pig and the Secret Ingredient Adventure\nPeppa: \"Hello, everyone! Today, we have a fun story about something very interesting. Do you know what tumors are? They're tiny little things that can grow inside our bodies, and they need a special ingredient to grow, just like how our cakes need sugar to be sweet!\"\nGeorge: \"Ooh, what's the special ingredient, Peppa?\"\nPeppa: \"It's called acetate! Imagine you're making a magic potion, and acetate is the magic dust that makes it work. Tumors use this acetate to grow bigger and stronger.\"\nGeorge: \"Wow, that's interesting! But how do they get this magic dust?\"\nPeppa: \"Great question, George! Scientists found out that there's a little helper called ACSS2. This helper is like a tiny worker that grabs the acetate from our food and gives it to the tumors.\"\nGeorge: \"And what did the scientists do next?\"\nPeppa: \"They did a clever experiment with mice! They took away the ACSS2 helper from the mice. Guess what happened? The tumors couldn't get their special ingredient and stayed small!\"\nGeorge: \"That's great news, Peppa! What does it mean for us?\"\nPeppa: \"It means scientists can make medicines to block the ACSS2 helper. It's like finding a way to stop the magic potion from working, so the tumors can't grow anymore. Isn't that cool?\"\nGeorge: \"That's amazing, Peppa! Scientists are like real-life superheroes!\"\nPeppa: \"They sure are, George! And that's our fun story for today, everyone. Remember, science can do wonderful things!\"\nGeorge: \"Bye-bye, everyone!\"\nPeppa and George: \"Oink, oink!\""} 
    # summary = {1: "Peppa Pig and the Secret Ingredient Adventure", 2: "Peppa: \"Hello, everyone! Today, we have a fun story about something very interesting. Do you know what tumors are? They're tiny little things that can grow inside our bodies, and they need a special ingredient to grow, just like how our cakes need sugar to be sweet!\"", 3: "George: \"Ooh, what's the special ingredient, Peppa?\"", 4: "Peppa: \"It's called acetate! Imagine you're making a magic potion, and acetate is the magic dust that makes it work. Tumors use this acetate to grow bigger and stronger.\"", 5: "George: \"Wow, that's interesting! But how do they get this magic dust?\"", 6: "Peppa: \"Great question, George! Scientists found out that there's a little helper called ACSS2. This helper is like a tiny worker that grabs the acetate from our food and gives it to the tumors.\"", 7: "George: \"And what did the scientists do next?\"", 8: "Peppa: \"They did a clever experiment with mice! They took away the ACSS2 helper from the mice. Guess what happened? The tumors couldn't get their special ingredient and stayed small!\"", 9: "George: \"That's great news, Peppa! What does it mean for us?\"", 10: "Peppa: \"It means scientists can make medicines to block the ACSS2 helper. It's like finding a way to stop the magic potion from working, so the tumors can't grow anymore. Isn't that cool?\"", 11: "George: \"That's amazing, Peppa! Scientists are like real-life superheroes!\"", 12: "Peppa: \"They sure are, George! And that's our fun story for today, everyone. Remember, science can do wonderful things!\"", 13: "George: \"Bye-bye, everyone!\"", 14: "Peppa and George: \"Oink, oink!\""} 
    summary = {1: 'Peppa: Daddy, what is this paper about?',
 2: "Daddy Pig: Well, Peppa, it's about a new model called the Transformer.",
 3: 'Peppa: Ooh, a transformer like in my toy box?',
 4: 'Daddy Pig: Not exactly, Peppa. This is a special kind of model that helps with translating words from one language to another.',
 5: 'Peppa: Oh, like when I say "Bonjour" and you say "Hello"?',
 6: 'Daddy Pig: Yes, just like that! But this new model is very clever because it uses something called "attention" instead of other types of models like "recurrent" or "convolutional" ones.',
 7: "Peppa: Attention? What's that?",
 8: "Daddy Pig: It's like when you pay close attention to something. The Transformer uses attention to help it understand words better.",
 9: 'Peppa: Oh, I see. But why is this model better than the others?',
 10: "Daddy Pig: Well, Peppa, it's because the Transformer is like a big tower with lots of layers. And each layer helps the model to understand words in a different way.",
 11: "Peppa: Wow, that's amazing! How does it work?",
 12: "Daddy Pig: The Transformer has a special function that takes a question and a group of words and gives an answer. It's like when you ask me a question and I give you an answer.",
 13: 'Peppa: I get it! But what\'s this thing called "multi-head attention"?',
 14: "Daddy Pig: That's a big word, Peppa. It means that the Transformer can pay attention to lots of things at the same time, like you playing with lots of toys at once.",
 15: 'Peppa: Oh, that sounds like fun! But what about these "positional encodings"?',
 16: 'Daddy Pig: Those are like little codes that tell the Transformer where each word is in a sentence. Like when you put your toys in a certain order, you know which one comes first and which one comes next.',
 17: 'Peppa: I see! And what did the Transformer do in the end?',
 18: 'Daddy Pig: Well, Peppa, it did very well! It was able to translate words better than any other model before it. And it can also do other things like understanding sentences and even learning new things.',
 19: "Peppa: Wow, that's so cool! Can I play with the Transformer too?",
 20: 'Daddy Pig: Of course, Peppa! The code for the Transformer is available for everyone to use. Maybe one day you can make your own model too!',
 21: 'Peppa: Yay, that would be fun! Thanks, Daddy!',
 22: "Daddy Pig: You're welcome, Peppa. Now let's go play with your toys."}
    dalle_api_key_file = 'dalle_api_key.txt'
    images = image.generage_image_dalle(summary, dalle_api_key_file)


    # Module 3 - Generate BGM
    # Input: 'summary'
    # Output: 'bgms' - a Python dictionary
    # Description: Each value of 'bgms' is the background music associated with each page of the final product.
    #              indexed by numbers starting from 1.
    summary = {1: "Peppa Pig and the Secret Ingredient Adventure", 2: "Peppa: \"Hello, everyone! Today, we have a fun story about something very interesting. Do you know what tumors are? They're tiny little things that can grow inside our bodies, and they need a special ingredient to grow, just like how our cakes need sugar to be sweet!\"", 3: "George: \"Ooh, what's the special ingredient, Peppa?\"", 4: "Peppa: \"It's called acetate! Imagine you're making a magic potion, and acetate is the magic dust that makes it work. Tumors use this acetate to grow bigger and stronger.\"", 5: "George: \"Wow, that's interesting! But how do they get this magic dust?\"", 6: "Peppa: \"Great question, George! Scientists found out that there's a little helper called ACSS2. This helper is like a tiny worker that grabs the acetate from our food and gives it to the tumors.\"", 7: "George: \"And what did the scientists do next?\"", 8: "Peppa: \"They did a clever experiment with mice! They took away the ACSS2 helper from the mice. Guess what happened? The tumors couldn't get their special ingredient and stayed small!\"", 9: "George: \"That's great news, Peppa! What does it mean for us?\"", 10: "Peppa: \"It means scientists can make medicines to block the ACSS2 helper. It's like finding a way to stop the magic potion from working, so the tumors can't grow anymore. Isn't that cool?\"", 11: "George: \"That's amazing, Peppa! Scientists are like real-life superheroes!\"", 12: "Peppa: \"They sure are, George! And that's our fun story for today, everyone. Remember, science can do wonderful things!\"", 13: "George: \"Bye-bye, everyone!\"", 14: "Peppa and George: \"Oink, oink!\""} 
    dalle_api_key_file = 'dalle_api_key.txt'
    bgms = music.generate_bgm(summary, dalle_api_key_file)


    # Module 4 - Text to Speech
    # Input: 'summary'
    # Output: 'Speech' - a Python dictionary
    # Description: Each value of 'images' is the read-out of text on each page of the final product, 
    #              indexed by numbers starting from 1.
    voices = voice.generate_voice(summary, other inputs)


    # combine multimedia 
    # Task is to combine all 4 elements (summary, image, bgm, voice) by aligning the keys of each Py dictionary.
    # e.g. The first page of our final product consist of summary[1], images[1], bgms[1] and voices[1].
    # maybe this step can be done in the GUI?
    final_product = result.Combine(summary)
    return jsonify(result = final_product)

if __name__ == '__main__':
    app.run(debug=True)