import logging
import os
from typing import Callable, Optional
import webdataset as wds

from webdataset.tariterators import tarfile_to_samples

logger = logging.getLogger("dinov2")


def log_and_continue(exn):
    logger.warning(f"Handling webdataset error ({repr(exn)}). Ignoring and continuing.")
    return True


def create_webdataset(
    *,
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    resample_shards: bool = True,
    **kwargs,
):
    """
    Creates a WebDataset instance for DINOv2 self-supervised learning,
    using the wds.DataPipeline constructor and setting a length for compatibility.
    """
    logger.info(f"Creating a WebDataset from {root}")

    pipeline = []

    # Using ResampledShards for "infinite" training as it's common for SSL
    pipeline.extend(
        [
            wds.ResampledShards(root),
        ]
    )

    # Split the work *after* shuffling and resampling the shards
    pipeline.extend(
        [
            wds.split_by_node,
            wds.split_by_worker,
        ]
    )

    pipeline.append(tarfile_to_samples(handler=log_and_continue))
    pipeline.append(wds.shuffle(5128, initial=512))
    pipeline.append(wds.decode("pil", handler=log_and_continue))

    def map_sample(sample):
        image_key = next(
            (key for key in sample if key.endswith((".jpg", ".jpeg", ".png"))), None
        )
        if image_key is None:
            return None
        image = sample[image_key]
        target = -1
        return image, target

    pipeline.append(wds.map(map_sample))
    pipeline.append(wds.select(lambda x: x is not None))

    if transform is not None and target_transform is not None:
        pipeline.append(wds.map_tuple(transform, target_transform))

    dataset = wds.DataPipeline(*pipeline)

    # --- 5. Set the dataset length for the trainer ---
    try:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        num_samples_per_process = 2_700_000 // world_size

        # ==============================================================================
        # THIS IS THE FIX: Use .with_length() so that len(dataset) works as expected.
        # ==============================================================================
        dataset = dataset.with_length(num_samples_per_process)

        # The log messages from your previous attempt show this part works, so we keep it.
        logger.info(
            f"Estimated {num_samples_per_process} samples per process (world size: {world_size})."
        )

    except (ValueError, TypeError, ZeroDivisionError):
        logger.warning(
            "Could not determine world size. Using total sample count for length."
        )
        dataset = dataset.with_length(2_700_000)

    logger.info("WebDataset created successfully using the DataPipeline pattern.")
    return dataset
