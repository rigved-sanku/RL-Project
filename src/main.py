# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Train and Eval PPOClipAgent in the Mujoco environments.

All hyperparameters come from the PPO paper
https://arxiv.org/abs/1707.06347.pdf
"""
import os


from absl import logging
import tensorflow.compat.v2 as tf
from train_eval_lib import train_eval

import os

def find_max_exp_number(root_dir):
    existing_folders = [folder for folder in os.listdir(root_dir) if folder.startswith("exp")]

    if not existing_folders:
        return 0

    exp_numbers = [int(folder[3:]) for folder in existing_folders]

    return max(exp_numbers)

def create_exp_folder(root_dir):
    max_exp_number = find_max_exp_number(root_dir)

    new_exp_number = max_exp_number + 1 if max_exp_number is not None else 1

    new_folder_name = f"exp{new_exp_number:04d}"

    new_folder_path = os.path.join(root_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path

def main():
    logging.set_verbosity(logging.INFO)
    tf.enable_v2_behavior()

    train_output_path = "/home/ramu/Personal/RL-Project/train_output/"
    root_dir = create_exp_folder(train_output_path)
    evaluation_file_path = f"{root_dir}/evaluations/"
    os.makedirs(evaluation_file_path, exist_ok=True)
    
    num_iterations = 50
    reverb_port = 13412
    eval_interval = 5

    train_eval(
        root_dir,
        evaluation_file_path,
        # Training params
        num_iterations=num_iterations,
        # Replay params
        reverb_port=reverb_port,
        # Others
        eval_interval=eval_interval,
    )


if __name__ == '__main__':
    main()
