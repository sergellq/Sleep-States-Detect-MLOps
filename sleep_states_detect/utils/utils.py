import glob
import os
from pathlib import Path

import torch


def check_files_exist(data_dir: Path | str, file_list: list[Path | str]) -> bool:
    """Checks if all files from the provided list exist in the specified directory.

    Args:
        data_dir (Path): The directory in which to check for the files.
        file_list (List[str]): Filenames to check for existence in the given directory.

    Returns:
        bool: `True` if all files exist in the directory, `False` otherwise.
    """
    # Проверяем, существует ли папка
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Папка {data_dir} не существует.")
        return False

    # Проверяем, существуют ли все файлы
    return all((data_dir / file).exists() for file in file_list)


def check_gpu_memory(required_memory: float, device: torch.device = None):
    """Checks if there is enough memory available on the GPU.

    Args:
        required_memory (float): The required memory in megabytes (MB).
        device (torch.device, optional): The device (default is 'cuda').

    Returns:
        bool: True if there is enough memory, False otherwise.
    """
    if device is None:
        device = torch.device("cuda")

    total_memory = (
        torch.cuda.get_device_properties(device).total_memory / 1024**2
    )  # MB
    free_memory = torch.cuda.memory_reserved(device) / 1024**2  # MB
    allocated_memory = torch.cuda.memory_allocated(device) / 1024**2  # MB

    print(f"Total memory: {total_memory} MB")
    print(f"Free memory: {free_memory} MB")
    print(f"Allocated memory: {allocated_memory} MB")

    available_memory = free_memory - allocated_memory
    if available_memory >= required_memory:
        print(f"Enough memory to run the model: {available_memory:.2f} MB available.")
        return True
    else:
        print(
            f"Not enough memory to run the model. Required: {required_memory:.2f} MB, "
            f"Available: {available_memory:.2f} MB."
        )
        return False


def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    """
    Find and return the path to the most recently modified .ckpt file in a directory.

    Args:
        checkpoint_dir (str): Path to the directory containing checkpoint files.

    Returns:
        str: Path to the latest checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint files are found in the directory.
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint files found.")
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint
