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
DEFAULT_RULES_PATH = "\\extensions\\traceryprompts\\mainrules.json"
RULES_DIR_PATH = "\\extensions\\traceryprompts\\rules"



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
        if not self.json_loaded:
            print("Ui calling load with " + str(len(self.rules_dict)) + " keys already loaded.")
            self.load_json()

        if len(self.rules_dict) == 0:
            print(f"\nInvalid grammar. No rules specified.")
            gr.Markdown("Error: Could not read rules.json!")
            return [ True ]
        
        gr.Column()
        gr.Markdown("# Ultimate Randomizer of Ultimate Fun")
        gr.Markdown("With Tracery Prompts as the active script you can use the key of any of the lists below in place of tokens and they will be randomly replaced at generation time.")
        gr.Markdown("## Example:")
        gr.Markdown(">best quality, 1girl, mature female, \#hair_colour\#, \#hairstyle\#, very long hair, \#iris_colour\#, wearing a white (\#dress\#:1.2) and #female_clothing#")
        gr.Markdown("## Notes:")
        gr.Markdown("- Check the prompt in your final image. If you see any of the keys in the promt surrounded by double parenthesis like ((hair_color)), then that key was not recognised and you probably spelt it wrong or forgot to include surrounding \# \n- These dropdowns don't do anything. They're just here to help you reference the keys and explore the lists for ideas.")
        gr.Markdown("")
        gr.Markdown("## Reference:")

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
            showp = gr.Checkbox(label="Show prompt in console", value=True)

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
            with open(scripts.basedir() + RULES_DIR_PATH + "\\" + f) as data_file:
                new_dict:dict = new_rules.copy()
                new_dict.update(json.load(data_file))
                new_rules = new_dict
        
        final_rules:dict = main_rules.copy()
        final_rules.update(new_rules)
        self.rules_dict = final_rules

        
        print("Done. Loaded: " + str(len(self.rules_dict)) + " entries.")
        
