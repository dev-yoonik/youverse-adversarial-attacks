import os
import shutil
from typing import Callable
from .attack_dataset import BaseDatasetAttacker, BaseDatasetAttackerMultithreaded
import tqdm
import random
import cv2
import logging


class LFWBenchmarkAttackerMultithreaded(BaseDatasetAttackerMultithreaded):
    """
    Attacker to create custom benchmark format.
    """
    def __init__(self,
                 dataset_root: str,
                 attack_name: str,
                 pairs_txt_path: str,
                 num_threads: int = 4,
                 random_percentage: float = None,
                 create_attack: Callable = None,
                 attack_dataset_root: str = None,
                 logging_path: str = "multiprocessing.log",
                 **kwargs):

        super().__init__(dataset_root=dataset_root,
                         attack_name=attack_name,
                         num_threads=num_threads,
                         random_percentage=random_percentage,
                         create_attack=create_attack,
                         create_attack_directory=False,
                         targeted_list=None,
                         attack_dataset_root=attack_dataset_root,
                         logging_path=logging_path)

        self.pairs, self.labels = self.read_pairs_file(pairs_txt_path, dataset_root)

        self.attack_paths, self.target_paths = self.create_benchmark_directory(self.pairs,
                                                                               self.labels,
                                                                               self.attack_dataset_root)
        self.targeted_list = [True if x == 0 else False if x == 1 else None for x in self.labels]

        # Split dataset into num_threads attack_parts
        self.attack_parts = self.split_dataset(self.attack_paths, num_threads)
        self.target_parts = self.split_dataset(self.target_paths, num_threads)
        self.targeted_parts = self.split_dataset(self.targeted_list, num_threads)

        # Create attack
        self.create_attack = create_attack

    def create_benchmark_directory(self, pairs, labels, attack_dataset_root):
        """
        Creates benchmark directory
        :param pairs: list of tuples with pairs of images
        :param labels: list of labels corresponding to the pairs
        :param attack_dataset_root: path to the attack dataset
        :return: images to be attacked.
        """
        bonafide_paths = []
        target_paths = []
        self.logger.debug(f"Creating attack dataset {attack_dataset_root}")
        os.makedirs(attack_dataset_root, exist_ok=True)
        for i, (pair, label) in enumerate(zip(pairs, labels)):
            pair_dir_name = f"pair_{i}_label_{label}"
            os.makedirs(os.path.join(attack_dataset_root, pair_dir_name), exist_ok=True)

            for j, image_path in enumerate(pair):
                # COPY IMAGE
                image_name = os.path.basename(image_path)
                new_image_path = os.path.join(attack_dataset_root, pair_dir_name, f"im_{j}_{image_name}")
                shutil.copy(image_path, new_image_path)

                if j == 0:
                    bonafide_paths.append(new_image_path)
                if j == 1:
                    target_paths.append(new_image_path)

        return bonafide_paths, target_paths

    @staticmethod
    def read_pairs_file(pairs_file_path, images_dir):
        image_pairs = []
        labels = []

        with open(pairs_file_path, 'r') as file:
            # Read the number of sets and pairs
            num_sets, pairs_per_set = map(int, file.readline().strip().split())

            for _ in range(num_sets):
                # Matched pairs (same identity)
                for _ in range(pairs_per_set):
                    line = file.readline().strip().split()
                    name, n1, n2 = line[0], int(line[1]), int(line[2])
                    img1_path = os.path.join(images_dir, f"{name}/{name}_{n1:04}.jpg")
                    img2_path = os.path.join(images_dir, f"{name}/{name}_{n2:04}.jpg")

                    image_pairs.append((img1_path, img2_path))
                    labels.append(1)

                # Mismatched pairs (different identities)
                for _ in range(pairs_per_set):
                    line = file.readline().strip().split()
                    name1, n1, name2, n2 = line[0], int(line[1]), line[2], int(line[3])
                    img1_path = os.path.join(images_dir, f"{name1}/{name1}_{n1:04}.jpg")
                    img2_path = os.path.join(images_dir, f"{name2}/{name2}_{n2:04}.jpg")

                    image_pairs.append((img1_path, img2_path))
                    labels.append(0)

        return image_pairs, labels

    def attack_list(self,
                    num_thread: int,
                    dataset_root: str,
                    attack_dataset_root: str,
                    list_of_attack_paths: list,
                    create_attack: Callable,
                    list_of_target_paths: list = None,
                    list_of_targeted_bools: list = None,
                    q=None):
        """
                Attack all images in the dataset and save them in the attack dataset.
                :return: None
                """
        """Attack all images in the dataset and save them in the attack dataset."""
        # Set up process-specific logger
        process_logger = self.setup_process_logger(q, num_thread)

        # Create attack instance
        attack = create_attack()
        if hasattr(attack, 'logger'):
            attack.logger = process_logger

        index = 0
        process_logger.debug(f"Starting attacks in process {num_thread}")

        for path in tqdm.tqdm(list_of_attack_paths, desc=f"Thread {num_thread}"):

            # Read image
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

            # Read Target
            if list_of_target_paths is not None:
                target = cv2.cvtColor(cv2.imread(list_of_target_paths[index]), cv2.COLOR_BGR2RGB)
            else:
                target = None

            if list_of_targeted_bools is not None:
                targeted = list_of_targeted_bools[index]
            else:
                targeted = None

            # Attack
            process_logger.debug(f"Attacking {path}")
            image = attack(image=image, target=target, targeted=targeted)["image"]

            # Delete old image
            if os.path.exists(path):
                os.remove(path)

            # Save
            cv2.imwrite(path.replace(".jpg", ".png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            process_logger.debug(f"Saved {path}")
            index += 1
