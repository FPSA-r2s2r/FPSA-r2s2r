from pathlib import Path
import argparse
import json
import shutil
import gc
import numpy as np
import imageio.v2 as imageio
import zarr
import concurrent.futures
import multiprocessing
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer


def read_video_to_numpy(video_path):
    """
    Decode one mp4 into uint8 numpy array.

    Compared with frames.append(...) + np.stack(...), this tries to pre-allocate
    the final array when imageio can report frame count, avoiding an extra large
    temporary list + stack copy.
    """
    video_path = Path(video_path)
    reader = imageio.get_reader(str(video_path))

    try:
        try:
            n_frames = int(reader.count_frames())
        except Exception:
            n_frames = None

        # Best path: preallocate once, then fill frame by frame.
        if n_frames is not None and n_frames > 0:
            arr = None
            count = 0
            for i, frame in enumerate(reader):
                frame = np.asarray(frame, dtype=np.uint8)
                if arr is None:
                    arr = np.empty((n_frames, *frame.shape), dtype=np.uint8)
                if i >= n_frames:
                    # Very rare fallback if count_frames() under-reports.
                    extra = [frame]
                    for rest in reader:
                        extra.append(np.asarray(rest, dtype=np.uint8))
                    return np.concatenate([arr[:i], np.stack(extra, axis=0)], axis=0)
                arr[i] = frame
                count = i + 1

            if arr is None:
                raise RuntimeError(f"empty video: {video_path}")
            return arr[:count]

        # Fallback: unknown frame count.
        frames = [np.asarray(frame, dtype=np.uint8) for frame in reader]
        if len(frames) == 0:
            raise RuntimeError(f"empty video: {video_path}")
        return np.stack(frames, axis=0)

    finally:
        reader.close()


def _load_npz_array(npz, key, dtype=np.float32):
    """Load lowdim arrays as float32 to reduce RAM and zarr size."""
    arr = np.asarray(npz[key])
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def load_one_episode(args):
    ep_dir, only_success, use_eye_in_hand = args
    ep_dir = Path(ep_dir)

    meta_path = ep_dir / "episode_meta.json"
    if not meta_path.exists():
        return None, f"[skip] {ep_dir.name}: missing episode_meta.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    success = bool(meta.get("success", False))
    if only_success and not success:
        return None, f"[skip] {ep_dir.name}: success=False"

    agentview_path = ep_dir / "agentview.mp4"
    lowdim_path = ep_dir / "lowdim.npz"

    required_paths = [agentview_path, lowdim_path]
    if use_eye_in_hand:
        eye_in_hand_path = ep_dir / "eye_in_hand.mp4"
        required_paths.append(eye_in_hand_path)

    if not all(path.exists() for path in required_paths):
        return None, f"[skip] {ep_dir.name}: missing data files"

    try:
        with np.load(lowdim_path) as lowdim:
            episode = {
                "agentview_image": read_video_to_numpy(agentview_path),
                "robot0_eef_pos": _load_npz_array(lowdim, "robot0_eef_pos"),
                "robot0_eef_quat": _load_npz_array(lowdim, "robot0_eef_quat"),
                "robot0_gripper_qpos": _load_npz_array(lowdim, "robot0_gripper_qpos"),
                "action": _load_npz_array(lowdim, "action"),
            }

        if use_eye_in_hand:
            episode["robot0_eye_in_hand_image"] = read_video_to_numpy(eye_in_hand_path)

        T = episode["action"].shape[0]
        keys_to_check = [
            "agentview_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        if use_eye_in_hand:
            keys_to_check.append("robot0_eye_in_hand_image")

        if not all(episode[k].shape[0] == T for k in keys_to_check):
            return None, f"[skip] {ep_dir.name}: length mismatch"

        msg = (
            f"[add] {ep_dir.name}: len={T}, "
            f"agentview={episode['agentview_image'].shape}, "
        )
        if use_eye_in_hand:
            msg += f"eye_in_hand={episode['robot0_eye_in_hand_image'].shape}, "
        msg += f"action={episode['action'].shape}, success={success}"
        return episode, msg

    except Exception as e:
        return None, f"[skip] {ep_dir.name}: error={repr(e)}"


def _make_disk_store(out_path):
    """Diffusion Policy commonly uses zarr v2, where DirectoryStore is available."""
    out_path = Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.DirectoryStore(str(out_path))


def _iter_loaded_parallel_bounded(tasks, num_workers, prefetch):
    """
    Keep only a small bounded number of in-flight episodes.
    This prevents many decoded video arrays from waiting in completed futures.
    """
    task_iter = iter(tasks)
    futures = set()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(min(prefetch, len(tasks))):
            try:
                futures.add(executor.submit(load_one_episode, next(task_iter)))
            except StopIteration:
                break

        with tqdm(total=len(tasks), desc="Loading episodes") as pbar:
            while futures:
                done, futures = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    yield future.result()
                    pbar.update(1)

                    try:
                        futures.add(executor.submit(load_one_episode, next(task_iter)))
                    except StopIteration:
                        pass


def collect_dataset_from_phydomain_parallel(
    data_root,
    out_path,
    only_success=True,
    use_eye_in_hand=False,
    num_workers=1,
    prefetch_factor=1,
    print_each=False,
):
    """
    Memory-efficient converter for Diffusion Policy replay buffer.

    Main changes from the original version:
    1. Writes zarr directly to disk instead of zarr.MemoryStore().
    2. Defaults to num_workers=1 to avoid many mp4 videos decoded at once.
    3. If multiprocessing is used, keeps only a bounded number of in-flight jobs.
    4. Casts lowdim/action arrays to float32.

    Recommended settings:
      - Lowest RAM: num_workers=1
      - Faster but still safe: num_workers=2 or 4, prefetch_factor=1
    """
    data_root = Path(data_root) / "episodes"
    episode_dirs = sorted(data_root.glob("episode_*"))

    if num_workers is None:
        # Do NOT use cpu_count() // 2 by default for video conversion.
        # Too many workers means too many decoded videos resident in RAM.
        num_workers = 1
    num_workers = max(1, int(num_workers))
    prefetch = max(1, int(num_workers) * int(prefetch_factor))

    # Critical fix: disk-backed zarr. Do not use zarr.MemoryStore() here.
    store = _make_disk_store(out_path)
    buffer = ReplayBuffer.create_empty_zarr(storage=store)

    num_total = len(episode_dirs)
    num_added = 0
    num_skipped = 0

    tasks = [(ep_dir, only_success, use_eye_in_hand) for ep_dir in episode_dirs]

    if num_workers == 1:
        iterator = (load_one_episode(task) for task in tasks)
        iterator = tqdm(iterator, total=len(tasks), desc="Loading episodes")
    else:
        iterator = _iter_loaded_parallel_bounded(tasks, num_workers, prefetch)

    for episode, msg in iterator:
        if print_each:
            print(msg)

        if episode is None:
            num_skipped += 1
            continue

        buffer.add_episode(episode)
        num_added += 1

        # Release this episode before loading/writing the next one.
        del episode
        gc.collect()

    print(f"\nDone. total={num_total}, added={num_added}, skipped={num_skipped}")
    print(f"zarr written directly to: {out_path}")
    return buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PhyDomain episodes to Diffusion Policy zarr replay buffer."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the dataset root directory.",
    )

    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output zarr path.",
    )

    parser.add_argument(
        "--use_eye_in_hand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use eye-in-hand camera. Default: True.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=25,
        help="Number of parallel workers. Default: 25.",
    )

    args = parser.parse_args()

    buffer = collect_dataset_from_phydomain_parallel(
        data_root=args.data_root,
        out_path=args.out_path,
        only_success=True,
        use_eye_in_hand=args.use_eye_in_hand,
        num_workers=args.num_workers,
        prefetch_factor=20,
        print_each=False,
    )

    print("done")

    buffer = ReplayBuffer.create_from_path(args.out_path, mode="r")
    print(buffer.meta)
    print(buffer.data.keys())
    print(buffer["action"].shape, buffer["action"].dtype)