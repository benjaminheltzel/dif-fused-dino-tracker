import argparse
from tqdm import tqdm
import pandas as pd
import os
import pickle
from eval.metrics import compute_tapvid_metrics_for_video, compute_badja_metrics_for_video

def eval_single_video(args):
    """
    Evaluate TAP-Vid metrics on a single video.
    
    Args:
        args: Command line arguments containing:
            - dataset_root_dir: Root directory containing video subdirectories
            - benchmark_pickle_path: Path to benchmark pickle file
            - out_file: Where to save results
            - dataset_type: Either "tapvid" or "BADJA"
            - video_idx: Index of the video to evaluate
    """
    # Load benchmark data
    benchmark_data = pickle.load(open(args.benchmark_pickle_path, "rb"))
    
    # Convert video index to string format as used in directory names
    video_idx_str = str(args.video_idx)
    video_dir = os.path.join(args.dataset_root_dir, video_idx_str)
    trajectories_dir = os.path.join(video_dir, "trajectories")
    occlusions_dir = os.path.join(video_dir, "occlusions")
    
    if not os.path.exists(video_dir):
        raise ValueError(f"Video directory {video_dir} does not exist")

    # Compute metrics for single video
    if args.dataset_type == "tapvid":
        metrics = compute_tapvid_metrics_for_video(
            model_trajectories_dir=trajectories_dir, 
            model_occ_pred_dir=occlusions_dir,
            video_idx=args.video_idx,
            benchmark_data=benchmark_data,
            pred_video_sizes=[854, 476]
        )
    elif args.dataset_type == "BADJA":
        metrics = compute_badja_metrics_for_video(
            model_trajectories_dir=trajectories_dir, 
            video_idx=args.video_idx,
            benchmark_data=benchmark_data,
            pred_video_sizes=[854, 476]
        )
    else:
        raise ValueError("Invalid dataset type. Must be either tapvid or BADJA")
    
    metrics["video_idx"] = args.video_idx
    
    # Convert to DataFrame and save
    metrics_df = pd.DataFrame([metrics])
    metrics_df.set_index('video_idx', inplace=True)
    metrics_df.to_csv(args.out_file)
    
    print(f"Metrics for video {args.video_idx}:")
    print(metrics_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root-dir", default="./dataset/davis_256", type=str)
    parser.add_argument("--benchmark-pickle-path", default="./dataset/davis.pkl", type=str) 
    parser.add_argument("--out-file", default="./single_video_metrics.csv", type=str)
    parser.add_argument("--dataset-type", default="tapvid", type=str, help="Dataset type: tapvid or BADJA")
    parser.add_argument("--video-idx", type=int, required=True, help="Index of video to evaluate")
    
    args = parser.parse_args()
    eval_single_video(args)

# example usage:
# python eval_single_video.py --dataset-root-dir ./dataset/tapvid-davis \
#                       --benchmark-pickle-path ./tapvid/tapvid_davis_data_strided.pkl \
#                       --out-file ./single_video_metrics.csv \
#                       --dataset-type tapvid \
#                       --video-idx 29