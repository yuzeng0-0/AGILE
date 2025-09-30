import numpy as np
import copy
from verl.workers.agent.tool_envs import ToolBase
from typing import Optional, List, Dict, Any
from PIL import Image
import re
import json
from math import ceil, floor
from .utils import base_python, diff_dict, filter_dict
import copy

class PuzzleToolBox(ToolBase):
    name = "puzzle_toolbox"
    
    def __init__(self, _name, _desc, _params, **kwargs):
        super().__init__(
            name=self.name,
        )
        self.chatml_history = []
        self.multi_modal_data = None  # To store the current image being processed
        self.old_env_variables = {}

    def execute(self, action_string: str, **kwargs) -> tuple:
        code_blocks = re.findall(r'<code>(.*?)</code>', action_string, re.DOTALL)
        answer_blocks = re.findall(r'<answer>(.*?)</answer>', action_string, re.DOTALL)
        try:
            if len(code_blocks)==1 and len(answer_blocks)==0:
                new_env_variables = copy.deepcopy(self.old_env_variables)
                # exec(base_python + "\n".join([f"{key} = {repr(value)}" for key, value in self.old_env_variables.items()]) + code_blocks[0], new_env_variables)
                exec(base_python + code_blocks[0], new_env_variables)
                new_env_variables = filter_dict(new_env_variables, "image")
                diff = diff_dict(new_env_variables, self.old_env_variables)
                assert len(diff)==1, "The python code can only perform one image block exchange at a time."
                self.old_env_variables = new_env_variables
                current_image = list(diff.values())[0]
                assert isinstance(current_image, Image.Image), f"Expected a PIL Image, but got {type(current_image)}"
                obs = {
                    "prompt": "\n<|im_start|>user\n" + f"The code is executed successfully. `{list(diff.keys())[0]}` is" +"<image>" + "<|im_end|>\n<|im_start|>assistant\n",
                    "multi_modal_data": {"image": [current_image]}
                }
                reward = 0.0  # Reward for successful tool call with correct JSON
                done = False
                info = {"status": "success"}
                print(f'[DEBUG] SUCCESS ACTION {action_string=}')

            elif len(code_blocks)==0 and len(answer_blocks)==1:
                return "", 0.0, True, {}
            
            else:
                return "", 0.0, True, {}
            
        except Exception as e:
            obs = "\n<|im_start|>user\n" + f"Error: {str(e)}" + "<|im_end|>\n<|im_start|>assistant\n"
            reward = 0.0  # No reward for failed execution
            done = False
            info = {"error": str(e), "status": "failed"}
        return obs, reward, done, info

    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        if len(self.multi_modal_data['image']) == 1:
            self.old_env_variables = {'raw_image': self.multi_modal_data['image'][0]}
        elif len(self.multi_modal_data['image']) == 5:
            self.old_env_variables = {'image_list': self.multi_modal_data['image'][:4],
                                      'reference_image': self.multi_modal_data['image'][4]}
        elif len(self.multi_modal_data['image']) == 4:
            self.old_env_variables = {'image_list': self.multi_modal_data['image']}
        else:
            raise ValueError("The input image exceeds the limit")
        assert 'image' in self.multi_modal_data.keys(), f'[ERROR] {origin_multi_modal_data=}'
        assert len(self.multi_modal_data['image']) > 0, f'[ERROR] {self.multi_modal_data["image"]=}'
