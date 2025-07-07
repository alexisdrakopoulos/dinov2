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
            return torch.zeros((3, 224, 224))  # Return a dummy tensor


def calculate_recall_at_k(retrieved_items, ground_truth_items, k):
    """Calculates if any of the top-k retrieved items are in the ground truth set."""
    return len(set(retrieved_items[:k]) & set(ground_truth_items)) > 0


# --- Main Evaluation Function ---


def run_retrieval_evaluation(model, cfg, iteration_info):
    """
    Runs a full image retrieval evaluation benchmark.
    1. Computes embeddings for all images in the benchmark set.
    2. Builds a FAISS index.
    3. Queries the index and calculates Recall@k.
    """
    # This evaluation should only run on the main process
    if not distributed.is_main_process():
        return

    # Use the teacher model for evaluation
    eval_model = model.teacher
    with FSDP.summon_full_params(eval_model, writeback=False, rank0_only=True):
        # --- 2. Main Process Guard ---
        # Now that the model is ready on Rank 0, we can have only Rank 0 do the work.
        if distributed.is_main_process():
            eval_cfg = cfg.eval.retrieval
            benchmark_dir = eval_cfg.benchmark_dir
            all_paths_file = os.path.join(benchmark_dir, "all_paths.json")
            ground_truth_file = os.path.join(benchmark_dir, "ground_truth.json")

            logger.info("--- Starting Image Retrieval Evaluation ---")
            logger.info(f"Loading assets from: {benchmark_dir}")

            with open(all_paths_file, "r") as f:
                all_paths = json.load(f)
            with open(ground_truth_file, "r") as f:
                ground_truth = json.load(f)

            # --- 2. Setup Dataset and Dataloader ---
            # Define a standard evaluation transform
            # Use the same resolution as DINOv2's global crops
            eval_transform = transforms.Compose(
                [
                    transforms.Resize(
                        cfg.crops.global_crops_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(cfg.crops.global_crops_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
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

            # --- 3. Compute Embeddings ---
            logger.info("Computing embeddings for the benchmark dataset...")
            all_embeddings = []
            logger.info("Setting model to evaluation mode...")
            eval_model.eval()
            with torch.no_grad():
                logger.info("Processing images in batches...")
                for batch in tqdm(dataloader, desc="Computing Embeddings"):
                    images = batch.to("cuda", non_blocking=True)
                    # DINOv2's forward pass returns a dict. We want the CLS token feature.
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        features = eval_model.backbone(images)
                    features /= features.norm(dim=-1, keepdim=True)  # Normalize
                    all_embeddings.append(features.cpu().numpy())

            master_embeddings = np.vstack(all_embeddings)

            # --- 4. FAISS Indexing and Search ---
            logger.info("Building FAISS index...")
            d = master_embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(master_embeddings.astype("float32"))
            logger.info(f"FAISS index built with {index.ntotal} vectors.")

            path_to_idx = {path: i for i, path in enumerate(all_paths)}
            query_paths = list(ground_truth.keys())
            query_indices = [path_to_idx[p] for p in query_paths]
            query_embeddings = master_embeddings[query_indices]

            # We search for max(RECALL_KS) + 1 because the top result is often the query itself.
            recall_ks = eval_cfg.recall_ks
            search_k = max(recall_ks) + 1
            logger.info(f"Performing batched search for top {search_k} results...")
            _, top_k_indices = index.search(
                query_embeddings.astype("float32"), search_k
            )

            # --- 5. Calculate Metrics ---
            recall_scores = {k: 0 for k in recall_ks}
            num_queries = len(query_paths)

            for i in tqdm(range(num_queries), desc="Calculating Recall@k"):
                query_path = query_paths[i]
                retrieved_indices = top_k_indices[i]
                retrieved_paths = [
                    all_paths[idx] for idx in retrieved_indices if idx != -1
                ]

                # Filter out the query itself from the results
                filtered_results = [p for p in retrieved_paths if p != query_path]

                true_positives = ground_truth[query_path]
                for k in recall_ks:
                    if calculate_recall_at_k(filtered_results, true_positives, k):
                        recall_scores[k] += 1

            # --- 6. Log and Save Results ---
            results = {}
            logger.info("--- Retrieval Benchmark Results ---")
            for k in recall_ks:
                score = (recall_scores[k] / num_queries) * 100
                logger.info(f"Recall@{k:<3}: {score:.2f}%")
                results[f"Recall@{k}"] = score
            logger.info("-----------------------------------")

            # Save results to a file for tracking
            eval_dir = os.path.join(cfg.train.output_dir, "eval", iteration_info)
            os.makedirs(eval_dir, exist_ok=True)
            results_path = os.path.join(eval_dir, "retrieval_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Retrieval results saved to {results_path}")

    logger.info(f"Rank {distributed.get_rank()} waiting at barrier.")
    torch.distributed.barrier()
    logger.info(f"Rank {distributed.get_rank()} passed barrier.")

    # Put the model back in training mode
    eval_model.train()
