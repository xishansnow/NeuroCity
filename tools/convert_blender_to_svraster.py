import os
import json
import shutil
import numpy as np
import argparse


def convert_blender_to_svraster(src_dir, dst_dir, split="train"):
    """
    Convert NeRF synthetic (blender) dataset to SVRasterDataset format.
    Args:
        src_dir: Source directory (e.g. .../nerf_synthetic/lego)
        dst_dir: Output directory (e.g. .../svraster_lego)
        split: train/test/val
    """
    os.makedirs(os.path.join(dst_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "poses"), exist_ok=True)

    # 1. Load transforms_*.json
    meta_path = os.path.join(src_dir, f"transforms_{split}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} not found")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # 2. Copy images and save poses
    for idx, frame in enumerate(meta["frames"]):
        # Blender file_path is relative, e.g. "./train/r_0"
        rel_img = frame["file_path"]
        if rel_img.startswith("./"):
            rel_img = rel_img[2:]
        src_img = os.path.join(src_dir, rel_img + ".png")
        dst_img = os.path.join(dst_dir, "images", f"image_{idx:03d}.png")
        if not os.path.exists(src_img):
            raise FileNotFoundError(f"Image not found: {src_img}")
        shutil.copy(src_img, dst_img)
        # Save pose
        pose = np.array(frame["transform_matrix"])
        pose_path = os.path.join(dst_dir, "poses", f"pose_{idx:03d}.txt")
        np.savetxt(pose_path, pose)

    # 3. Save intrinsics.txt
    H = meta.get("h", 800)
    W = meta.get("w", 800)
    camera_angle_x = meta["camera_angle_x"]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])
    np.savetxt(os.path.join(dst_dir, "intrinsics.txt"), K)
    print(f"Converted {len(meta['frames'])} images and poses to {dst_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NeRF synthetic (blender) dataset to SVRasterDataset format."
    )
    parser.add_argument(
        "--src", required=True, help="Source directory (e.g. .../nerf_synthetic/lego)"
    )
    parser.add_argument(
        "--dst", required=True, help="Destination directory (e.g. .../svraster_lego)"
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "test", "val"], help="Which split to convert"
    )
    args = parser.parse_args()
    convert_blender_to_svraster(args.src, args.dst, args.split)


if __name__ == "__main__":
    main()
