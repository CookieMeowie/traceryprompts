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

class Script(scripts.Script):  

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
        jsonpath = gr.Text(label="Rules JSON path", value=scripts.basedir() + "/extensions/traceryprompts/rules.json")
        showp = gr.Checkbox(label="Show prompt in console", value=True)
        return [ jsonpath, showp ]

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.
    def run(self, p, jsonpath, showp):
        # Load rules.json
        with open(jsonpath) as data_file:
            rules_dict = json.load(data_file)

        if len(rules_dict) == 0:
            print(f"\nInvalid grammar. No rules specified.")
            return

        # Add main prompt in as main
        rules_dict["main"] = p.prompt

        # Finalise tracery
        grammar = tracery.Grammar(rules_dict)

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