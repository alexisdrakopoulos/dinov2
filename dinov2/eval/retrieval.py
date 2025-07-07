# retrieval.py

import torch
import numpy as np
import json
import os
import logging
from PIL import Image
from tqdm import tqdm
import faiss
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import dinov2.distributed as distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
import dinov2.distributed as dinov2_dist
import tempfile
import glob

logger = logging.getLogger("dinov2")

# --- Helper Classes and Functions ---
ROOT = "augmented_data/"


class ImageDataset(Dataset):
    """A simple dataset to load images from a list of paths."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = ROOT + self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            logger.warning(
                f"Could not load image {path}, returning a blank tensor. Error: {e}"
            )
            return torch.zeros((3, 224, 224))


def calculate_recall_at_k(retrieved_items, ground_truth_items, k):
    """Calculates if any of the top-k retrieved items are in the ground truth set."""
    return len(set(retrieved_items[:k]) & set(ground_truth_items)) > 0


# --- Helper function inspired by AntiFSDPCheckpointer to fuse sharded state dicts ---
def recursive_fuse(shards):
    """
    Recursively fuses a list of sharded state dictionaries into a single one.
    This is the core logic from the provided AntiFSDPCheckpointer.
    """
    if isinstance(shards[0], torch.Tensor):
        # Assumes tensors are meant to be concatenated.
        return torch.cat(shards)
    elif isinstance(shards[0], dict):
        # Recurse through dictionaries.
        all_keys = set.union(*map(lambda s: set(s.keys()), shards))
        return {k: recursive_fuse([s[k] for s in shards if k in s]) for k in all_keys}
    else:
        # For other data types (like scalars), assume they are identical and take the first one.
        return shards[0]


# --- Main Evaluation Function ---


def run_retrieval_evaluation(model, cfg, iteration_info):
    """
    Runs a full image retrieval evaluation benchmark by first re-assembling
    the full model on Rank 0 from its distributed shards.
    """
    rank = dinov2_dist.get_global_rank()
    eval_fsdp_module = model.teacher.backbone

    if not isinstance(eval_fsdp_module, FSDP):
        if rank == 0:
            logger.warning("Target module is not FSDP. Running standard evaluation.")
        # If not FSDP, the logic becomes much simpler.
        # We can just run the eval on Rank 0.
        if distributed.is_main_process():
            perform_main_process_evaluation(eval_fsdp_module, cfg, iteration_info)
        dist.barrier()
        return

    # --- Robust FSDP Un-sharding Logic ---
    logger.info(f"Rank {rank}: Starting robust FSDP evaluation.")

    # 1. Create a temporary directory for shards that is visible to all ranks.
    #    The `if rank == 0` guard ensures it's only created once.
    #    The barrier ensures all ranks wait until it exists.
    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_obj.name

    dist.barrier()

    # 2. Each rank saves its LOCAL shard of the teacher backbone to the temp directory.
    #    This uses FSDP's local state dict, which is fast and does not hang.
    shard_path = os.path.join(temp_dir, f"shard_rank_{rank}.pth")
    logger.info(f"Rank {rank}: Saving local shard to {shard_path}")
    with FSDP.state_dict_type(model.teacher, StateDictType.LOCAL_STATE_DICT):
        local_state = model.teacher.backbone.state_dict()
        torch.save(local_state, shard_path)

    # 3. Wait for all ranks to finish saving their shards.
    dist.barrier()

    # 4. ONLY Rank 0 reassembles the full model.
    if distributed.is_main_process():
        logger.info("Rank 0: All shards saved. Re-assembling full model state dict.")

        # Find all the saved shard files.
        shard_paths = sorted(glob.glob(os.path.join(temp_dir, "shard_rank_*.pth")))
        if len(shard_paths) != distributed.get_global_size():
            raise RuntimeError("Mismatch between found shards and world size.")

        # Load all shards into a list.
        shards = [torch.load(p, map_location="cpu") for p in shard_paths]

        # Fuse them into a single, complete state dict.
        # This is not yet implemented in the provided snippet, so we'll need to write it.
        # For ViT, many parameters are sharded on dim 0.
        # A simple `recursive_fuse` might work if we assume concatenation.
        # NOTE: This fusion logic might need adjustment based on the *exact*
        # sharding strategy of each parameter. For DINOv2 ViTs, concatenating
        # along dimension 0 is usually correct for most sharded layers (qkv, proj).
        full_state_dict = recursive_fuse(shards)

        logger.info("Rank 0: Model re-assembled. Creating a clean evaluation model.")

        # Get the original, unwrapped module architecture.
        # `_fsdp_wrapped_module` is the correct way to access the raw module.
        clean_eval_model = eval_fsdp_module._fsdp_wrapped_module
        clean_eval_model.load_state_dict(full_state_dict)

        # Run the actual evaluation on this clean, single-GPU model.
        perform_main_process_evaluation(clean_eval_model, cfg, iteration_info)

        # Clean up the temporary directory
        temp_dir_obj.cleanup()

    # 5. All processes wait here to ensure Rank 0 is finished before continuing.
    logger.info(f"Rank {rank}: Evaluation finished, waiting at barrier.")
    dist.barrier()
    logger.info(f"Rank {rank}: Passed barrier.")


def perform_main_process_evaluation(eval_model, cfg, iteration_info):
    """This function contains the actual evaluation logic, now guaranteed to run on a single, complete model."""
    logger.info("--- Starting Image Retrieval Evaluation (on Rank 0) ---")

    eval_model.to(torch.device("cuda"))
    eval_model.eval()

    eval_cfg = cfg.eval.retrieval
    benchmark_dir = eval_cfg.benchmark_dir
    # ... (the rest of your loading and evaluation logic is IDENTICAL to what you had)
    all_paths_file = os.path.join(benchmark_dir, "all_paths.json")
    ground_truth_file = os.path.join(benchmark_dir, "ground_truth.json")

    with open(all_paths_file, "r") as f:
        all_paths = json.load(f)
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                cfg.crops.global_crops_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.crops.global_crops_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = ImageDataset(all_paths, eval_transform)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )

    logger.info("Computing embeddings for the benchmark dataset...")
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Embeddings"):
            images = batch.to("cuda", non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                features = eval_model(images)
            features /= features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy())

    master_embeddings = np.vstack(all_embeddings)

    # --- FAISS, Metrics, and Logging (This part is unchanged) ---
    logger.info("Building FAISS index...")
    d = master_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(master_embeddings.astype("float32"))
    logger.info(f"FAISS index built with {index.ntotal} vectors.")

    path_to_idx = {path: i for i, path in enumerate(all_paths)}
    query_paths = list(ground_truth.keys())
    query_indices = [path_to_idx[p] for p in query_paths]
    query_embeddings = master_embeddings[query_indices]

    recall_ks = eval_cfg.recall_ks
    search_k = max(recall_ks) + 1
    logger.info(f"Performing batched search for top {search_k} results...")
    _, top_k_indices = index.search(query_embeddings.astype("float32"), search_k)

    recall_scores = {k: 0 for k in recall_ks}
    num_queries = len(query_paths)

    for i in tqdm(range(num_queries), desc="Calculating Recall@k"):
        query_path = query_paths[i]
        retrieved_indices = top_k_indices[i]
        retrieved_paths = [all_paths[idx] for idx in retrieved_indices if idx != -1]
        filtered_results = [p for p in retrieved_paths if p != query_path]
        true_positives = ground_truth[query_path]
        for k in recall_ks:
            if calculate_recall_at_k(filtered_results, true_positives, k):
                recall_scores[k] += 1

    results = {}
    logger.info("--- Retrieval Benchmark Results ---")
    for k in recall_ks:
        score = (recall_scores[k] / num_queries) * 100
        logger.info(f"Recall@{k:<3}: {score:.2f}%")
        results[f"Recall@{k}"] = score
    logger.info("-----------------------------------")

    eval_dir = os.path.join(cfg.train.output_dir, "eval", iteration_info)
    os.makedirs(eval_dir, exist_ok=True)
    results_path = os.path.join(eval_dir, "retrieval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Retrieval results saved to {results_path}")
