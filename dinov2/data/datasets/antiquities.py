import logging
import os
from typing import Callable, Optional
import webdataset as wds

# We need these components to build the pipeline list
from webdataset.tariterators import tarfile_to_samples

logger = logging.getLogger("dinov2")


# This is a helper function to handle exceptions during processing, like OpenCLIP does.
def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logger.warning(f"Handling webdataset error ({repr(exn)}). Ignoring and continuing.")
    return True


def create_webdataset(
    *,
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    # Add a flag to easily control resampling
    resample_shards: bool = True,
    **kwargs,
):
    """
    Creates a WebDataset instance for DINOv2 self-supervised learning,
    using the wds.DataPipeline constructor for compatibility and clarity.
    """
    logger.info(f"Creating a WebDataset from {root}")

    # --- 1. Define the list of processing stages, like in OpenCLIP ---
    pipeline = []

    if resample_shards:
        # Use ResampledShards to continuously sample from the shard list.
        # This is great for "infinite" training.
        pipeline.extend(
            [
                wds.ResampledShards(root),
                # NOTE: With ResampledShards, splitting happens *after* a shard is chosen.
                # The splitters ensure that each worker process gets unique samples from within the shard.
                # DINOv2's original implementation implies this is the desired behavior for SSL.
                wds.split_by_worker,
            ]
        )
    else:
        # For sequential, non-resampled training (e.g., validation)
        pipeline.extend(
            [
                wds.SimpleShardList(root),
                wds.split_by_node,
                wds.split_by_worker,
                # You might want to shuffle the order of shards each epoch
                wds.shuffle(100, initial=10),
            ]
        )

    # --- 2. Add stages for processing the data within the shards ---

    # This stage opens the .tar files and yields individual files ({'__key__': '...', 'image.jpg': ...})
    # We use `handler=log_and_continue` to make it robust to corrupted tars.
    pipeline.append(tarfile_to_samples(handler=log_and_continue))

    # Shuffle the samples coming from the tars. This is the main shuffling buffer.
    pipeline.append(wds.shuffle(5128, initial=512))

    # Decode the image data
    pipeline.append(wds.decode("pil", handler=log_and_continue))

    # --- 3. Map the data to the format DINOv2 expects ---
    def map_sample(sample):
        # Find the image key robustly
        image_key = next(
            (key for key in sample if key.endswith((".jpg", ".jpeg", ".png"))), None
        )
        if image_key is None:
            # Skip this sample if no image is found
            return None
        image = sample[image_key]
        # Target is a placeholder for Self-Supervised Learning
        target = -1
        return image, target

    # Apply the map function and filter out any None samples (e.g., from decoding errors)
    pipeline.append(wds.map(map_sample))
    pipeline.append(wds.select(lambda x: x is not None))

    # Apply the DINOv2-specific data augmentations
    if transform is not None and target_transform is not None:
        logger.info("Applying DINOv2 transforms to the WebDataset pipeline.")
        pipeline.append(wds.map_tuple(transform, target_transform))

    # --- 4. Create the final dataset object from the pipeline list ---
    dataset = wds.DataPipeline(*pipeline)

    # --- 5. Set the dataset length for the trainer ---
    try:
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        # This is an estimate for the progress bar.
        num_samples_per_process = 2_700_000 // world_size
        dataset = dataset.with_epoch(num_samples_per_process)
        logger.info(
            f"Estimated {num_samples_per_process} samples per process (world size: {world_size})."
        )
    except (ValueError, TypeError, ZeroDivisionError):
        # Fallback if WORLD_SIZE is not set or invalid
        logger.warning(
            "Could not determine world size. Using total sample count for length."
        )
        dataset = dataset.with_length(2_700_000)

    logger.info("WebDataset created successfully using the DataPipeline pattern.")
    return dataset
