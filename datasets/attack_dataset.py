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


class BaseDatasetAttacker:
    logger = logging.getLogger(__name__)

    # TODO: ADD targeted option.
    # TODO: Add custom name for attack_dataset_root
    def __init__(self,
                 dataset_root: str,
                 attack: BaseAttack,
                 random_percentage: float = None,
                 **kwargs):

        # Attack
        self.attack = attack

        # Bona fide dataset
        self.dataset_root = dataset_root

        # Create attack dataset
        self.attack_dataset_root = self.dataset_root + "_" + attack.name
        self.create_attack_directory()

        # Get dataset paths
        self.bona_fide_paths = self.get_dataset_paths(self.dataset_root)

        # Create path subset if specified
        if random_percentage is not None:
            attack_paths = self.create_attack_subset(self.bona_fide_paths, random_percentage)
        else:
            attack_paths = self.bona_fide_paths

        self.attack_paths = attack_paths

    @staticmethod
    def create_attack_subset(path_list: list, percent: float):
        """
        Create a subset of the dataset.
        :param path_list: List of paths
        :param percent: Percentage of the dataset to use (between 0 and 1)
        :return: List of paths sampled from path_list
        """
        # Ensure the percentage is within a valid range
        if not (0 <= percent <= 1):
            raise ValueError("Percentage must be between 0 and 1")

        # Calculate the subset size
        subset_size = int(len(path_list) * percent)

        # Create random subset
        subset = random.sample(path_list, subset_size)

        return subset


    def create_attack_directory(self):
        """
        Create a copy of the folder structure of the bona fide dataset.
        :return: None
        """

        # Check if dataset exists
        if not os.path.exists(self.dataset_root):
            raise ValueError(f"Dataset root path {self.dataset_root} does not exist.")

        # Create copy of dataset
        self.logger.debug(f"Creating attack dataset {self.attack_dataset_root}")
        create_copy_dir(self.dataset_root, self.attack_dataset_root, keep_imgs=False)

    @staticmethod
    def get_dataset_paths(dataset_root: str):
        """
        Collect all paths of images in the dataset.
        :param dataset_root: The root path of the dataset.
        :return: A list of paths.
        """


        paths = []
        # Iterate over dataset and get each image
        for root, _, files in os.walk(dataset_root):
            for file in files:
                if file.split(".")[-1] in IMAGE_EXTENSIONS:
                    paths.append(os.path.join(root, file))

        return paths


    def attack_dataset(self):
        """
        Attack all images in the dataset and save them in the attack dataset.
        :return: None
        """
        self.logger.debug("Attacking dataset...")
        # Iterate over new dataset and attack each image
        for path in tqdm.tqdm(self.attack_paths):

            attack_path = path.replace(self.dataset_root, self.attack_dataset_root)

            # Read image
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

            # Attack
            self.logger.debug(f"Attacking {path}")
            image = self.attack(image=image)["image"]

            # Save
            cv2.imwrite(attack_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            self.logger.debug(f"Saved {attack_path}")


class BaseDatasetAttackerMultithreaded:

    def __init__(self,
                 dataset_root: str,
                 attack_name: str,
                 num_threads: int = 4,
                 random_percentage: float = None,
                 bona_fide_paths: list = None,
                 target_paths: list = None,
                 targeted_list: List[bool] = None,
                 create_attack: Callable = None,
                 create_attack_directory: bool = True,
                 attack_dataset_root: str = None,
                 logging_path: str = "multiprocess.log",
                 **kwargs):

        # Number of threads
        self.num_threads = num_threads

        # Bona fide dataset
        self.dataset_root = dataset_root

        # Create attack dataset
        if attack_dataset_root is not None:
            self.attack_dataset_root = attack_dataset_root
        else:
            self.attack_dataset_root = self.dataset_root + "_" + attack_name

        if create_attack_directory:
            self.create_attack_directory()

        # Get dataset paths
        if bona_fide_paths is None:
            self.bona_fide_paths = self.get_dataset_paths(self.dataset_root)
        else:
            self.bona_fide_paths = bona_fide_paths

        self.target_paths = target_paths
        self.targeted_list = targeted_list

        # if random_percentage is not None:
        #     attack_paths = self.create_attack_subset(self.bona_fide_paths, random_percentage)
        # else:
        #     attack_paths = self.bona_fide_paths
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


    def configure_listener_handlers(self):
        """Configure and return handlers for the listener process"""
        # Create handlers
        file_handler = logging.FileHandler(self.logging_path)
        #console_handler = logging.StreamHandler()

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        #console_handler.setFormatter(formatter)

        return [file_handler]#, console_handler]


    def listener_process(self, q):
        """Listener process that handles log records from the queue"""
        # Set up handlers
        handlers = self.configure_listener_handlers()

        while True:
            try:
                record = q.get()
                if record is None:  # Sentinel value to stop the listener
                    break

                # Create logger for this record
                logger = logging.getLogger(record.name)
                logger.handlers = []  # Remove any existing handlers

                # Add handlers to logger
                for handler in handlers:
                    logger.addHandler(handler)

                # Handle the record
                logger.handle(record)

            except Exception as e:
                import sys, traceback
                print(f"Error in logging listener: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        # Clean up handlers
        for handler in handlers:
            handler.close()

    @staticmethod
    def create_attack_subset(path_list: list, percent: float):
        """
        Create a subset of the dataset.
        :param path_list: List of paths
        :param percent: Percentage of the dataset to use (between 0 and 1)
        :return: List of paths sampled from path_list
        """
        # Ensure the percentage is within a valid range
        if not (0 <= percent <= 1):
            raise ValueError("Percentage must be between 0 and 1")

        # Calculate the subset size
        subset_size = int(len(path_list) * percent)

        # Create random subset
        subset = random.sample(path_list, subset_size)

        return subset

    def create_attack_directory(self):
        """
        Create a copy of the folder structure of the bona fide dataset.
        :return: None
        """

        # Check if dataset exists
        if not os.path.exists(self.dataset_root):
            raise ValueError(f"Dataset root path {self.dataset_root} does not exist.")

        # Create copy of dataset
        self.logger.debug(f"Creating attack dataset {self.attack_dataset_root}")
        create_copy_dir(self.dataset_root, self.attack_dataset_root, keep_imgs=False)

    @staticmethod
    def get_dataset_paths(dataset_root: str):
        """
        Collect all paths of images in the dataset.
        :param dataset_root: The root path of the dataset.
        :return: A list of paths.
        """


        paths = []
        # Iterate over dataset and get each image
        for root, _, files in os.walk(dataset_root):
            for file in files:
                if file.split(".")[-1] in IMAGE_EXTENSIONS:
                    paths.append(os.path.join(root, file))

        return paths

    @staticmethod
    def split_dataset(paths: list, num_splits: int):

        # Split full list into num_split attack_parts
        part_size = len(paths) // num_splits
        parts = [paths[i:i + part_size] for i in range(0, len(paths), part_size)]

        return parts

    def setup_process_logger(self, q, num_thread):
        """Configure logger for a worker process"""
        logger = logging.getLogger(f"attack_{num_thread}")
        logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        logger.handlers = []

        # Add queue handler
        queue_handler = QueueHandler(q)
        logger.addHandler(queue_handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

        return logger

    def attack_list(self,
                    num_thread: int,
                    dataset_root: str,
                    attack_dataset_root: str,
                    list_of_attack_paths: list,
                    create_attack: Callable,
                    list_of_target_paths: list = None,
                    list_of_targeted_bools: list = None,
                    q=None):
        """Attack all images in the dataset and save them in the attack dataset."""
        # Set up process-specific logger
        process_logger = self.setup_process_logger(q, num_thread)

        # Create attack instance
        attack = create_attack()
        if hasattr(attack, 'logger'):
            attack.logger = process_logger

        index = 0
        process_logger.debug(f"Starting attacks in process {num_thread}")

        for path in tqdm.tqdm(list_of_attack_paths, desc=f"Process {num_thread}"):
            attack_path = path.replace(dataset_root, attack_dataset_root)
            # TODO: Add for other file types
            attack_path.replace(".jpg", ".png")
            if os.path.exists(attack_path):
                continue

            try:
                # Read image
                image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

                # Handle target if needed
                target = None
                if list_of_target_paths is not None:
                    target = cv2.cvtColor(cv2.imread(list_of_target_paths[index]), cv2.COLOR_BGR2RGB)

                targeted = list_of_targeted_bools[index] if list_of_targeted_bools is not None else None

                # Attack
                process_logger.debug(f"Attacking {path}")
                image = attack(image=image, target=target, targeted=targeted)["image"]

                # Save
                os.makedirs(os.path.dirname(attack_path), exist_ok=True)
                cv2.imwrite(attack_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                process_logger.debug(f"Saved {attack_path}")

            except Exception as e:
                process_logger.error(f"Error processing {path}: {str(e)}")

            index += 1

    def attack_dataset(self):
        processes = []

        # Create logging queue
        log_queue = multiprocessing.Queue(-1)

        # Start listener process
        listener = multiprocessing.Process(target=self.listener_process, args=(log_queue,))
        listener.start()
        self.logger.debug("Started logging listener process")

        try:
            # Create and start worker processes
            for i in range(self.num_threads):
                target_parts = self.target_parts[i] if hasattr(self, 'target_parts') else None
                targeted_parts = self.targeted_parts[i] if hasattr(self, 'targeted_parts') else None
                process = multiprocessing.Process(
                    target=self.attack_list,
                    args=(i, self.dataset_root, self.attack_dataset_root,
                          self.attack_parts[i], self.create_attack,
                          target_parts, targeted_parts, log_queue)
                )
                process.start()
                processes.append(process)

            # Wait for all processes to complete
            for process in processes:
                process.join()

        finally:
            # Ensure cleanup happens even if there's an error
            log_queue.put(None)  # Signal listener to stop
            listener.join()

            # Close the queue
            log_queue.close()
            log_queue.join_thread()