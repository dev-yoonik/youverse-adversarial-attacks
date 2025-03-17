from abc import ABC
import os
import cv2
import torch
from typing import Callable, List
from fr_attacks.base_attack import BaseAttack
import tqdm
import random
from multiprocessing import Process
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import IMAGE_EXTENSIONS, create_copy_dir
import multiprocessing
from logging.handlers import QueueHandler
from .attack_dataset import BaseDatasetAttacker, BaseDatasetAttackerMultithreaded


class CelebAAttackerMultithreaded(BaseDatasetAttackerMultithreaded):

    def __init__(self,
                 dataset_root: str,
                 attack_label_list: str,
                 num_threads: int = 4,
                 create_attack: Callable = None,
                 create_attack_directory: bool = True,
                 attack_dataset_root: str = None,
                 logging_path: str = "multiprocess.log",
                 **kwargs):

        super().__init__(dataset_root=dataset_root,
                         attack_name="",
                         num_threads=num_threads,
                         random_percentage=None,
                         create_attack=create_attack,
                         create_attack_directory=False,
                         targeted_list=None,
                         attack_dataset_root=attack_dataset_root,
                         logging_path=logging_path)

        # Number of threads
        self.num_threads = num_threads

        # Bona fide dataset
        self.dataset_root = dataset_root

        # Create attack dataset
        if attack_dataset_root is not None:
            self.attack_dataset_root = attack_dataset_root
        else:
            self.attack_dataset_root = self.dataset_root + "_attacked"

        if create_attack_directory:
            self.create_attack_directory()

        self.bona_fide_paths, self.target_paths, self.targeted_list = self.read_labels(label_path=attack_label_list)

        attack_paths = self.bona_fide_paths

        # Split dataset into num_threads attack_parts
        self.attack_parts = self.split_dataset(attack_paths, num_threads)
        if self.target_paths is not None:
            self.target_parts = self.split_dataset(self.target_paths, num_threads)
        else:
            self.target_parts = None

        if self.targeted_list is not None:
            self.targeted_parts = self.split_dataset(self.targeted_list, num_threads)
        else:
            self.targeted_parts = None

        # Create attack
        self.create_attack = create_attack
        self.logger = logging.getLogger(__name__)
        self.logging_path = logging_path

    @staticmethod
    def read_labels(label_path: str):

        # Initialize the three lists.
        bonafide_list = []
        target_list = []
        targeted_list = []

        # Read and parse the file.
        with open(label_path, "r") as f:
            lines = f.readlines()

        # Check if there's a header; if so, skip it.
        if lines and lines[0].startswith("original_image"):
            lines = lines[1:]

        # Process each line.
        for line in lines:
            # Expected format per line:
            # original_image original_identity attack_type attack_image attack_id
            parts = line.strip().split()
            if not parts:
                continue
            # Unpack the first five columns (ignore any extra columns if present)
            original_image, original_identity, attack_type, attack_image, attack_id = parts[:5]

            # Append to lists.
            bonafide_list.append(original_image)
            target_list.append(attack_image)

            # Set True if impersonation, False if evasion.
            targeted_list.append(attack_type.lower() == "impersonation")

        return bonafide_list, target_list, targeted_list