import os
import shutil

def copy_folders_with_multiple_files(source_root, destination_root):
    """
    Copies folders with more than one file from the source_root to the destination_root.

    Parameters:
        source_root (str): Path to the root source folder.
        destination_root (str): Path to the root destination folder.
    """
    if not os.path.exists(source_root):
        raise ValueError(f"Source root folder '{source_root}' does not exist.")

    # Ensure the destination root exists
    os.makedirs(destination_root, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(source_root):
        # Calculate the relative path from the source root
        relative_path = os.path.relpath(dirpath, source_root)

        # Skip the root folder itself
        if relative_path == ".":
            continue

        # Only process folders with more than one file
        if len(filenames) > 1:
            # Determine the destination folder path
            dest_dir = os.path.join(destination_root, relative_path)

            # Create the folder structure in the destination root
            os.makedirs(dest_dir, exist_ok=True)

            # Copy the files to the destination folder
            for filename in filenames:
                src_file = os.path.join(dirpath, filename)
                dest_file = os.path.join(dest_dir, filename)
                shutil.copy2(src_file, dest_file)

if __name__ == "__main__":
    # Example usage
    source_folder = r"D:\datasets\adversarial\lfw-deepfunneled_aligned\lfw-deepfunneled"
    destination_folder = r"D:\datasets\adversarial\lfw-deepfunneled_aligned\lfw-deepfunneled-more-imgs-per-id"

    try:
        copy_folders_with_multiple_files(source_folder, destination_folder)
        print(f"Folders with more than one file have been copied to '{destination_folder}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
