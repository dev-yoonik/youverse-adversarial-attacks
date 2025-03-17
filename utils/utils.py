import torch
import os
import shutil
import numpy as np
def prep_data_for_torch(data, add_batch_dim=False):
    """
    Convert numpy or tensor data to tensor format [C,H,W] and range [0,1].
    Assumes input is in [H,W,C] for numpy and [C,H,W] for tensor.
    """
    if isinstance(data, np.ndarray):
        if data.dtype == np.uint8:
            data = data.astype(np.float32)
            data /= 255.0
        # If float assume already in range 0-1 and don't divide by 255 else assume uint8 and divide
        data = torch.from_numpy(data).permute(2, 0, 1)


    elif isinstance(data, torch.Tensor):
        assert data.ndim == 3, "Tensor must have 3 dimensions."  # [C,H,W]
        assert 0 <= data.min() <= 1 and 0 <= data.max() <= 1, "Tensor should be in 0-1 range."
    else:
        raise TypeError("Input data must be numpy.ndarray or torch.Tensor.")

    if add_batch_dim:
        data = data.unsqueeze(0)
    return data

@staticmethod
def prep_data_for_cv2(data):
    """
    Data out should be in cv2 RGB format [0-1]. As such if the input is not in this format it should be converted.
    """
    if isinstance(data, torch.Tensor):
        if data.ndim == 4:
            data = data.squeeze(0)
        assert data.ndim == 3, "Tensor must have 3 dimensions."
        data = data.cpu().permute(1, 2, 0).squeeze(0).detach().numpy().astype(np.float32)

    assert data.shape[-1] in [1,3,4] and data.ndim == 3, "Wrong shape of data. Should be in [H,W,C]."
    assert data.max() <= 1 and data.min() >= 0 and data.dtype == np.float32, "Wrong range/type of data. Should be in float 0-1 range."
    return (data * 255.0).astype(np.uint8)


IMAGE_EXTENSIONS = ["jpg", "JPG", "png", "PNG", "ppm", "PPM", "tif", "TIF", "tiff", "TIFF", "bmp", "BMP", "jpeg", "JPEG"]
VIDEO_EXTENSIONS = ["mov", "MOV", "mp4", "MP4", "avi", "AVI"]


def ig_all_files(root, files):
    # All files
    return [f for f in files if os.path.isfile(os.path.join(root, f))]


def ig_only_images(root, files):
    # Only Image files
    return [f for f in files if os.path.isfile(os.path.join(root, f)) and f[-3:] in IMAGE_EXTENSIONS]


def ig_only_videos(root, files):
    # Only Video files
    return [f for f in files if os.path.isfile(os.path.join(root, f)) and f[-3:] in VIDEO_EXTENSIONS]


def ig_images_and_videos(root, files):
    # Image and Video files
    return [f for f in files if
            os.path.isfile(os.path.join(root, f)) and (f[-3:] in VIDEO_EXTENSIONS or f[-3:] in IMAGE_EXTENSIONS)]


def ig_all_except_img(root, files):
    # All except image files
    return [f for f in files if os.path.isfile(os.path.join(root, f)) and not f[-3:] in IMAGE_EXTENSIONS]


def ig_all_except_vid(root, files):
    # All except video files
    return [f for f in files if os.path.isfile(os.path.join(root, f)) and not f[-3:] in VIDEO_EXTENSIONS]


def create_copy_dir(root, new_root, keep_files: bool = False, keep_imgs: bool = False,
                    keep_videos: bool = False):
    """
    Creates a copy of a dataset structure
    :param root: root of the dataset
    :param new_root: root for copy
    :param keep_files: keep all files that are not image or video
    :param keep_imgs: keep images
    :param keep_videos: keep videos
    :return: None
    """
    if keep_files and keep_imgs and keep_videos:
        # Include all files
        ig_f = None
    elif not keep_files and not keep_imgs and not keep_videos:
        # Reject all files
        ig_f = ig_all_files
    elif keep_files and not keep_imgs and not keep_videos:
        # Include just other files
        ig_f = ig_images_and_videos
    elif not keep_files and keep_imgs and not keep_videos:
        # Include just images
        ig_f = ig_all_except_img
    elif not keep_files and not keep_imgs and keep_videos:
        # Include just video
        ig_f = ig_all_except_vid
    else:
        raise NotImplementedError

    try:
        shutil.copytree(root, new_root, ignore=ig_f)
    except FileExistsError:
        pass
