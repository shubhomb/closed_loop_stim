import os
import re
import numpy as np
import argparse
import zipfile

def load_experiment_data(experiment_dir):
    all_data = {}
    for subdir in sorted(os.listdir(experiment_dir)):
        data_dir = os.path.join(experiment_dir, subdir)
        if not os.path.isdir(data_dir):
            continue
        data_path = os.path.join(data_dir, "session_data.npz")
        data = np.load(data_path, allow_pickle=True)
        for k in data.files:
            arr = np.asarray(data[k])
            if arr.ndim == 1:      # (n_blocks,) -> (1, n_blocks)
                arr = arr[None, :]
            elif arr.ndim == 2:    # (n_blocks, d) -> (1, n_blocks, d)
                arr = arr[None, :, :]

            if k not in all_data:
                all_data[k] = arr
            else:
                all_data[k] = np.concatenate([all_data[k], arr], axis=0)

    return all_data

def collect_train_blocks(experiment_dir, field_name=None):
    """
    Find files named train_block_{i}_data.npz inside each session subdir.
    For each block index i, concatenate the requested field (or all fields if field_name is None)
    across sessions along axis=0. Returns a dict mapping field -> { str(i) : concatenated_array }.
    """
    block_pattern = re.compile(r"train_block_(\d+)_data\.npz$")
    # mapping: block_idx (str) -> list of dicts (npz file contents per session)
    blocks_per_index = {}

    for subdir in sorted(os.listdir(experiment_dir)):
        subpath = os.path.join(experiment_dir, subdir)
        if not os.path.isdir(subpath):
            continue
        # find matching files in this subdir
        for fname in os.listdir(subpath):
            m = block_pattern.match(fname)
            if not m:
                continue
            block_idx = m.group(1)   # keep as string for dict keys
            fpath = os.path.join(subpath, fname)
            try:
                npz = np.load(fpath, allow_pickle=True)
            except Exception as e:
                print(f"Warning: failed to load {fpath}: {e}")
                continue

            if block_idx not in blocks_per_index:
                blocks_per_index[block_idx] = []
            # store the loaded npz (Mapping of fields -> arrays)
            blocks_per_index[block_idx].append(npz)

    if not blocks_per_index:
        print("No test_block_*_data.npz files found in any session subdir.")
        return {}

    # decide which fields to process
    fields_to_process = set()
    if field_name:
        fields_to_process.add(field_name)
    else:
        # union of fields across first found npz files for each block
        for block_idx, npz_list in blocks_per_index.items():
            for npz in npz_list:
                for f in npz.files:
                    fields_to_process.add(f)

    result = {}  # mapping field -> dict(block_idx_str -> concatenated_array)
    for field in sorted(fields_to_process):
        field_blocks = {}
        for block_idx, npz_list in sorted(blocks_per_index.items(), key=lambda x: int(x[0])):
            arrays_for_block = []
            for npz in npz_list:
                if field not in npz.files:
                    # session doesn't include this field for this block; skip
                    continue
                arr = np.asarray(npz[field])
                # ensure consistent dimensionality: if necessary add leading axis so that
                # concatenation along axis=0 makes sense.
                # We assume arr's first axis indexes samples already; if arr is 0-d, wrap it.
                if arr.ndim == 0:
                    arr = arr[None]
                arrays_for_block.append(arr)

            if not arrays_for_block:
                # No session provided this field for this block; skip adding key
                print(f"Note: no data for field '{field}' in block {block_idx}; skipping that block.")
                continue

            try:
                concat = np.concatenate(arrays_for_block, axis=0)
            except Exception as e:
                # give debugging info and then try to stack if shapes differ only by leading axis
                print(f"Error concatenating field '{field}' for block {block_idx}: {e}")
                # re-raise or skip — here we skip this block but inform user
                continue

            field_blocks[block_idx] = concat

        if field_blocks:
            result[field] = field_blocks
        else:
            print(f"Warning: no blocks contained field '{field}'. No output will be created for this field.")

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to the experiment directory containing session subdirectories.")
    # parser.add_argument("--field", nargs="?", const=None, default=None, type=str,
    #                     help="Specific field to aggregate across train blocks. If not provided, will skip.")
    parser.add_argument("--save_zip", action="store_false",
                        help="If set, export all experiment data as a zip file.")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if not os.path.isdir(experiment_dir):
        raise ValueError(f"Experiment directory {experiment_dir} does not exist or is not a directory.")

    # load and save session_data aggregated file
    for subdir in sorted(os.listdir(experiment_dir)):
        subpath = os.path.join(experiment_dir, subdir)
        if not os.path.isdir(subpath):
            continue

        print(f"Processing session directory: {subpath}")

        session_data = load_experiment_data(subpath)
        if session_data:
            out_path = os.path.join(subpath, "experiment_data.npz")
            np.savez_compressed(out_path, **session_data)
            print(f"  Saved session experiment_data.npz to {out_path}")
        else:
            print(f"  No session_data.npz found inside {subpath}")

    # # collect and concatenate train_block_i_data.npz per block for requested field(s)
    # if args.field is None:
    #     print("No field specified for train block aggregation; skipping that step.")
    #     exit(0)
    # collected = collect_train_blocks(experiment_dir, field_name=args.field)
    # # collected: mapping field -> { block_idx_str : concatenated_array }

    # # Save outputs: one .npz per field (filename = <field>.npz in experiment_dir)
    # for field, blocks_dict in collected.items():
    #     if not blocks_dict:
    #         continue
    #     save_path = os.path.join(experiment_dir, f"{field}.npz")
    #     # keys must be strings; we already used block_idx as string
    #     np.savez_compressed(save_path, **blocks_dict)
    #     print(f"Saved aggregated field '{field}' with blocks {sorted(blocks_dict.keys(), key=int)} to {save_path}")
    if args.save_zip:
        zip_path = f"{experiment_dir.rstrip(os.sep)}.zip"
        print(f"Creating ZIP archive: {zip_path}")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for subdir in sorted(os.listdir(experiment_dir)):
                subpath = os.path.join(experiment_dir, subdir)
                if not os.path.isdir(subpath):
                    continue

                exp_file = os.path.join(subpath, "experiment_data.npz")
                if os.path.isfile(exp_file):
                    # Add to zip, storing the directory structure
                    arcname = os.path.join(subdir, "experiment_data.npz")
                    zf.write(exp_file, arcname)
                    print(f"  Added {arcname}")
                else:
                    print(f"  No experiment_data.npz found in {subdir}, skipping.")

        print("ZIP archive created successfully.")