import os
import sys

import faiss
import numpy as np
import torch
import parser

import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from loguru import logger
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import vpr_models
import visualizations

from test_dataset import TestDataset



def main(args):
    start_time = datetime.now()

    # logger.remove()
    # log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    # logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    # logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    # logger.add(log_dir / "debug.log", level="DEBUG")
    # print(" ".join(sys.argv))
    
    log_dir = Path(args.log_dir)
    print(f"Arguments: {args}")
    print(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    print(f"The outputs are being saved in {log_dir}")

    output_csv = os.path.join(log_dir, 'vpr_results.csv')
    os.makedirs(log_dir, exist_ok=True)

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    print(f"Testing on {test_ds}")

    with torch.inference_mode():
        # logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        # logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        for images, indices in tqdm(queries_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    if args.save_descriptors:
        # print(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    faiss_index.add(database_descriptors)
    # del database_descriptors, all_descriptors
    
    # prepare for manual ordering of images for similarity matrix
    query_path = args.queries_folder.replace('rgb', '')
    database_path = args.database_folder.replace('rgb', '')
    
    queries_rgb_txt_path = os.path.join(query_path, 'rgb.txt')
    database_rgb_txt_path = os.path.join(database_path, 'rgb.txt')

    query_df = pd.read_csv(queries_rgb_txt_path, sep=' ', header=None)
    db_df = pd.read_csv(database_rgb_txt_path, sep=' ', header=None)
    
    q_indices = []
    for image in query_df[1]:
        filename = os.path.join(query_path, image)
        if filename in test_ds.queries_paths:
            q_idx = test_ds.queries_paths.index(filename)
            q_indices.append(q_idx)
            
    db_indices = []
    for image in db_df[1]:
        filename = os.path.join(database_path, image)
        if filename in test_ds.database_paths:
            db_idx = test_ds.database_paths.index(filename)
            db_indices.append(db_idx)
    
    distance_matrix = faiss.pairwise_distances(database_descriptors, queries_descriptors)
    similarity_matrix = 1 / distance_matrix #np.linalg.inv(distance_matrix)
    similarity_matrix_sorted = np.zeros_like(similarity_matrix)
    for i, q_idx in enumerate(q_indices):
        for j, db_idx in enumerate(db_indices):
            similarity_matrix_sorted[j,i] = similarity_matrix[db_idx, q_idx]
            
    dist_outfile = os.path.join(log_dir, f'{args.method.lower()}_distance_matrix.npy')
    sim_outfile = os.path.join(log_dir, f'{args.method.lower()}_similarity_matrix.npy')
    np.save(dist_outfile, distance_matrix)
    np.save(sim_outfile, similarity_matrix_sorted)

    # logger.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

    # For each query, check if the predictions are correct
    if args.use_labels:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Divide by num_queries and multiply by 100, so the recalls are in percentages
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        print(recalls_str)

    if args.num_preds_to_save != 0:
        image_paths, visualisation_img_path = visualizations.save_preds(predictions[:, :args.num_preds_to_save], distances[:, :args.num_preds_to_save], test_ds,
                                log_dir, output_csv, args.save_only_wrong_preds, args.use_labels)
        if visualisation_img_path is not None:
            visualisation_img = Image.open(visualisation_img_path)
    
    end_time = datetime.now()
    
    print(f"ELAPSED: {end_time - start_time}")

if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
