import copy
import modules.scripts as scripts
import gradio as gr
import random
import requests
import tracery
import math
import os
import json
import gradio as gr
import torch
import argparse

from modules import images, processing, prompt_parser, scripts, shared
from modules.processing import Processed, process_images
from modules.shared import cmd_opts, opts, state


DEFAULT_RULES = "{\n\"origin\": \"#hello.capitalize#, #location#!\",\n\"hello\": [\"hello\", \"greetings\", \"howdy\", \"hey\"],\n\"location\": [\"world\", \"solor system\", \"galaxy\", \"universe\"]\n}"
DEFAULT_RULES_PATH = "/extensions/traceryprompts/mainrules.json"
RULES_DIR_PATH = "/extensions/traceryprompts/rules"



class Script(scripts.Script):  

    json_loaded:bool = False
    rules_dict:dict = {}

    def load_json(jsonpath):
        # Load rules.json
        with open(jsonpath) as data_file:
            rules_dict = json.load(data_file)
            return rules_dict

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Tracery Prompts"

# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.
    def show(self, is_img2img):
        return True

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):
        showp:bool = False

        with gr.Tab("Info"):

            with gr.Row():
                btn = gr.Button("Reload JSON")
                btn.click(self.do_btn_reload_json)

            if not self.json_loaded:
                print("Ui calling load with " + str(len(self.rules_dict)) + " keys already loaded.")
                self.load_json()

            if len(self.rules_dict) == 0:
                print(f"\nInvalid grammar. No rules specified.")
                gr.Markdown("Error: Could not read rules.json!") 
                return [ showp ]
            
            gr.Markdown("""

                # Tracery Prompts

                ## Description
                With TraceryPrompts as the active script you can use the name of any of the lists in the Reference tab, surrounded by # in the place of tokens and they will be randomly replaced at generation time. Check the example.

                Check the resulting image's prompt to see the results and check for any issues. Rules surrounded by ((double_parenthesis)) have not been recognised. Check your spelling.

                Currently only works in the positive prompt.

                ## Example Prompt
                > best quality, 1girl, mature female, #hair_colour#, #hairstyle#, very long hair, #iris_colour#, wearing a white (#dress#:1.2) and #female_clothing#

                ## Modifiers
                There are several modifiers available to help you.

                To use a modifier write a . after the rule name and then the name of the modifier. For example #hair_accent.ran1in2#

                **Randomizing**

                *Use these modifiers to only place certain rules in the prompt a pecentage of the time.*

                - *ran1in2* Will only add the specified rule 50% of the time
                - *ran1in3* Will only add the specified rule 33.3% of the time
                - *ran1in4* Will only add the specified rule 25% of the time
                - *ran1in8* Will only add the specified rule 12.5% of the time

                **Random Weights**

                *Randomly weight something. Try it with #art_movement.rw#*

                - *rw* Assigns a random weight between 0.0 and 2.0 to the result of this rule
                - *rwh* Assigns a random weight between 0.5 and 1.5 to the result of this rule. The H stands for half. 
                - *rwq* Assigns a random weight between 0.75 and 1.25 to the result of this rule. The Q stands for quarter.

                **Quick Weighting**

                *Quick weights can quickly give your rule any weight from 0.1 to 2.0 in increments of 0.1. For the sake of brevity they haven't all been listed.*
                - *w1-1* Give the result a weight of 1.1
                - *w0-1* Lowest available
                - *w2-0* Highest available

                ## Customising

                Navigate to the stable-diffusion-webui/extensions/traceryprompts/rules directory and you will find .json files containing the rules. You can add your own .json files here and they will be merged in to the rest of the rules.
                Use the **Reload JSON** button above to refresh any changes in the json without restarting Automatic1111.
                ## Notes
                - The dropdowns in the Reference tab don't do anything. They're just here to help you reference the keys and explore the lists for ideas.
                - It's very easy to forget the trailing #



                """)
        
        with gr.Tab("Reference"):

            keys = list(self.rules_dict.keys())
            # order
            keys.sort()
            i = 0
            while True:
                with gr.Row():
                    gr.Dropdown(self.rules_dict[keys[i]], label=keys[i])
                    i += 1
                    if i >= len(keys):
                        break
                    gr.Dropdown(self.rules_dict[keys[i]], label=keys[i])
                    i += 1
                    if i >= len(keys):
                        break
                    gr.Dropdown(self.rules_dict[keys[i]], label=keys[i])
                    i += 1
                    if i >= len(keys):
                        break
                    gr.Dropdown(self.rules_dict[keys[i]], label=keys[i])
                    i += 1
                    if i >= len(keys):
                        break
            
            with gr.Column():
                gr.Markdown("# Debug")
                showp = gr.Checkbox(value=False, label="Show prompt in console")

        return [ showp ]

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.
    def run(self, p, showp):
        # Add main prompt in as main
        self.rules_dict["main"] = p.prompt

        # Finalise tracery
        grammar = tracery.Grammar(self.rules_dict)
        grammar.add_modifiers(modifiers)

        # Prep for iterating
        BatchSize = p.batch_size
        IterCount = p.n_iter
        state.job_count = BatchSize * IterCount
        p.batch_size = 1
        p.n_iter = 1
        images = []
        infotexts = []

        # Iterate
        for y in range(BatchSize):
            for x in range(IterCount):
                # Edit prompt
                p.prompt = grammar.flatten("#main#")

                if showp:
                    print(f"\n\n" + p.prompt)

                # Finalise
                processed = process_images(p)
                images += processed.images
                infotexts += processed.infotexts

        p.batch_size = BatchSize
        p.n_iter = IterCount

        return Processed(p=p, images_list=images, info=infotexts[0], infotexts=infotexts)
    
    def load_json(self):
        self.json_loaded = True

        # Load main rules.json
        with open(scripts.basedir() + DEFAULT_RULES_PATH) as data_file:
            main_rules:dict = json.load(data_file)
        
        # Discover json files in the rules directory
        files = []
        for file in os.listdir(scripts.basedir() + RULES_DIR_PATH):
            if file.endswith(".json"):
                files.append(file)
        
        # Load each json
        new_rules:dict = {}
        for f in files:
            with open(scripts.basedir() + RULES_DIR_PATH + "/" + f) as data_file:
                new_dict:dict = new_rules.copy()
                new_dict.update(json.load(data_file))
                new_rules = new_dict
        
        final_rules:dict = main_rules.copy()
        final_rules.update(new_rules)
        self.rules_dict = final_rules

        
        print("Done. Loaded: " + str(len(self.rules_dict)) + " entries.")
        
    def do_btn_reload_json(self):
        self.load_json()



















def ran1in2(text, *params):
    return random.choice([text, ""])

def ran1in3(text, *params):
    return random.choice([text, "", ""])

def ran1in4(text, *params):
    return random.choice([text, "", "", ""])

def ran1in8(text, *params):
    return random.choice([text, "", "", "", "", "", "", ""])

def w1(text, *params):
    return "(" + text + ":1.1)"

def w2(text, *params):
    return "(" + text + ":1.2)"

def w3(text, *params):
    return "(" + text + ":1.3)"

def w4(text, *params):
    return "(" + text + ":1.4)"

def w5(text, *params):
    return "(" + text + ":1.5)"

def w6(text, *params):
    return "(" + text + ":1.6)"

def w7(text, *params):
    return "(" + text + ":1.7)"

def w8(text, *params):
    return "(" + text + ":1.8)"

def w9(text, *params):
    return "(" + text + ":1.9)"

def w10(text, *params):
    return "(" + text + ":2.0)"

def wn1(text, *params):
    return "(" + text + ":0.1)"

def wn2(text, *params):
    return "(" + text + ":0.2)"

def wn3(text, *params):
    return "(" + text + ":0.3)"

def wn4(text, *params):
    return "(" + text + ":0.4)"

def wn5(text, *params):
    return "(" + text + ":0.5)"

def wn6(text, *params):
    return "(" + text + ":0.6)"

def wn7(text, *params):
    return "(" + text + ":0.7)"

def wn8(text, *params):
    return "(" + text + ":0.8)"

def wn9(text, *params):
    return "(" + text + ":0.9)"

def rw(text, *params):
    return "(" + text + ":" + str(random.random() * 2.0) + ")"

def rwh(text, *params):
    return "(" + text + ":" + str(0.5 + random.random()) + ")"

def rwq(text, *params):
    return "(" + text + ":" + str(0.75 + random.random() * 0.5) + ")"

modifiers = {
    "ran1in2": ran1in2,
    "ran1in3": ran1in3,
    "ran1in4": ran1in4,
    "ran1in8": ran1in8,
    "w1-1": w1,
    "w1-2": w2,
    "w1-3": w3,
    "w1-4": w4,
    "w1-5": w5,
    "w1-6": w6,
    "w1-7": w7,
    "w1-8": w8,
    "w1-9": w9,
    "w2-0": w10,
    "w0-1": wn1,
    "w0-2": wn2,
    "w0-3": wn3,
    "w0-4": wn4,
    "w0-5": wn5,
    "w0-6": wn6,
    "w0-7": wn7,
    "w0-8": wn8,
    "w0-9": wn9,
    "rw": rw,
    "rwh": rwh
}