from pathlib import PurePosixPath
import shutil
import os


def copy_dir(src_dir, target_dir):
    """Copy the contents of a directory to another directory."""
    if not os.path.exists(src_dir):
        return
    target_dir = os.path.join(target_dir, src_dir)
    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(src_dir):
        shutil.copy(os.path.join(src_dir, filename), 
                    os.path.join(target_dir, filename))

def copy_media_files(block, block_vars, gallery_conf, **kwargs):
    target_file = block_vars["target_file"]
    target_dir = os.path.dirname(target_file)
    # create a video folder in the target directory
    copy_dir("videos", target_dir)
    copy_dir("figures", target_dir)

    # check if a thumbnail is defined
    env_vars = block_vars['example_globals']
    if "thumbnail" not in env_vars:
        return ""

    thumbnail = env_vars["thumbnail"]
    image_path_iterator = block_vars["image_path_iterator"]

    # copy the thumbnail to the image_path
    for filename, image_path in zip([thumbnail], image_path_iterator):
        image_path = PurePosixPath(image_path)

        # copy the file to image_path
        try:
            shutil.copy(filename, image_path)
        except FileNotFoundError:
            # if the file is not found, we skip it
            continue

    return ""