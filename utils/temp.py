import os
import shutil

import os


def remove_jpg_if_png_exists(folder):
    file_names = {}

    # Collect file names without extensions
    for file_name in os.listdir(folder):
        name_without_ext, ext = os.path.splitext(file_name)
        if ext.lower() in ['.jpg', '.png']:
            if name_without_ext not in file_names:
                file_names[name_without_ext] = []
            file_names[name_without_ext].append(ext.lower())

    # Remove JPGs if PNG exists
    for name, extensions in file_names.items():
        if '.png' in extensions and '.jpg' in extensions:
            jpg_path = os.path.join(folder, f"{name}.jpg")
            if os.path.exists(jpg_path):
                os.remove(jpg_path)
                print(f"Removed: {jpg_path}")


# Example usage
folder_path = r"D:\Challenge\img_align_celeba_aligned_attack\images"  # Change this to your folder path
remove_jpg_if_png_exists(folder_path)



def copy_unique_files(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get existing filenames in destination (ignoring extensions)
    existing_files = {os.path.splitext(f)[0] for f in os.listdir(destination_folder)}

    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)
        if os.path.isfile(file_path):
            name_without_ext, _ = os.path.splitext(file_name)
            if name_without_ext not in existing_files:
                shutil.copy(file_path, destination_folder)
                print(f"Copied: {file_name}")
            else:
                print(f"Skipped: {file_name} (Already exists)")


# Example usage
source_folder = r"D:\datasets\adversarial\img_align_celeba\img_align_celeba_aligned"  # Change this to your source folder
destination_folder = r"D:\Challenge\img_align_celeba_aligned_attack\images"  # Change this to your destination folder

copy_unique_files(source_folder, destination_folder)