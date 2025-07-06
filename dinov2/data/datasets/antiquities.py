import logging
from typing import Callable, Optional
import webdataset as wds

logger = logging.getLogger("dinov2")


def create_webdataset(
    *,
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    **kwargs,
):
    """
    Creates a WebDataset instance for DINOv2 self-supervised learning.

    Args:
        root (str): The path to the webdataset shards, supporting shell-like globbing.
            For example: "/path/to/shards/dataset-{000000..000127}.tar"
        transform (Optional[Callable]): A function/transform that takes in an image
            and returns a transformed version.
        target_transform (Optional[Callable]): A function/transform that takes in the
            target and transforms it. In DINOv2 SSL, this is usually a dummy function.
        **kwargs: Extra arguments (unused here but good for compatibility).
    """
    logger.info(f"Creating a WebDataset from {root}")

    # The `resampled=True` argument randomly samples from the shards with replacement.
    # This is great for "infinite" datasets and ensures good shuffling.
    dataset: wds.WebDataset = wds.WebDataset(root, resampled=True)

    # The processing pipeline for WebDataset.
    # 1. Shuffle shards and samples within shards.
    # 2. Decode the image from bytes to a PIL Image.
    # 3. For SSL, we only need the image. The target is discarded later.
    #    Here we map the dictionary to a tuple containing only the transformed image.
    #    DINOv2's `collate_fn` and training loop expect a (image, target) tuple.
    #    The provided `transform` is actually `DataAugmentationDINO`, which returns a
    #    dictionary of tensors. The `target_transform` is `lambda _: ()`.
    #    The dataset should yield a tuple `(image, target)`.
    #
    # We will let the `_make_sample_transform` wrapper in loaders.py handle the
    # separate `transform` and `target_transform`. Our job is to yield the
    # raw (PIL Image, empty_target) tuple.
    def map_sample(sample):
        # Assumes your webdataset has a key 'image.jpg' for the image
        image = sample["image.jpg"]
        # The target is a placeholder for SSL, DINOv2 will ignore it.
        target = -1  # or {} or None
        return image, target

    dataset: wds.WebDataset = (
        dataset.shuffle(5128)  # 1. Shuffle shards and samples
        .decode(
            "pil"
        )  # 2. Decode the image from bytes to PIL. The sample is still a dict.
        .map(map_sample)  # 3. Now, map the dict to the (image, target) tuple.
    )

    if transform is not None and target_transform is not None:
        logger.info("Applying DINOv2 transforms to the WebDataset pipeline.")
        dataset = dataset.map_tuple(transform, target_transform)

    total_samples = 2_700_000  # estimate of number of samples in the dataset
    dataset = dataset.with_length(total_samples)

    logger.info("WebDataset created successfully.")
    return dataset
