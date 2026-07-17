# #!/usr/bin/env python
# """Stage a PINNED list of Zero-To-CAD-1m UUIDs as STEP + CadQuery files.

# Drop-in replacement for the select_zero_to_cad.py step when the exact UUIDs
# to restore are already known (e.g. recovering a lost benchmark set).

# Usage:
#     python scripts/renderer/stage_pinned_ids.py \
#         --ids-file path/to/ids.txt \
#         --stage_dir experiments/stage_z2c_bench_old \
#         [--split validation]

# ids.txt: one UUID per line (blank lines and lines starting with '#' ignored).
# Output layout is identical to select_zero_to_cad.py so batch_dataset.py and
# build_fixtures.py can consume it unchanged.
# """
# import argparse
# import json
# import os

# from datasets import load_dataset


# def main():
#     parser = argparse.ArgumentParser(description=__doc__,
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("--ids-file", required=True,
#                         help="Text file with one UUID per line")
#     parser.add_argument("--stage_dir", required=True,
#                         help="Output directory (must not already exist)")
#     parser.add_argument("--split", default="validation",
#                         help="HF dataset split to search (default: validation)")
#     parser.add_argument("--no-code", action="store_true",
#                         help="Skip writing {uuid}.cadquery.py")
#     args = parser.parse_args()

#     # Read target IDs
#     with open(args.ids_file) as f:
#         target_ids = [
#             line.strip() for line in f
#             if line.strip() and not line.startswith("#")
#         ]
#     target_set = set(target_ids)
#     print(f"Target IDs: {len(target_set)}")

#     if os.path.isdir(args.stage_dir):
#         raise FileExistsError(f"{args.stage_dir!r} already exists. Remove it first.")
#     os.makedirs(args.stage_dir)

#     cols = ["uuid", "num_faces", "step_file"] + ([] if args.no_code else ["cadquery_file"])
#     print(f"Loading split={args.split} from ADSKAILab/Zero-To-CAD-1m …")
#     ds = load_dataset("ADSKAILab/Zero-To-CAD-1m", split=args.split).select_columns(cols)
#     print(f"Split loaded: {len(ds)} rows. Filtering {len(target_set)} UUIDs (batched) …", flush=True)

#     filtered = ds.filter(
#         lambda batch: [uid in target_set for uid in batch["uuid"]],
#         batched=True,
#         batch_size=1000,
#         num_proc=1,
#     )
#     print(f"  filter done: {len(filtered)} rows matched.")
#     found = {row["uuid"]: row for row in filtered}

#     missing = target_set - set(found)
#     if missing:
#         print(f"WARNING: {len(missing)} IDs not found in split={args.split}:")
#         for uid in sorted(missing):
#             print(f"  {uid}")

#     staged = 0
#     for uid in target_ids:
#         if uid not in found:
#             continue
#         row = found[uid]
#         sf = row["step_file"]
#         if not sf:
#             print(f"  SKIP {uid}: empty step_file")
#             continue
#         with open(os.path.join(args.stage_dir, uid + ".step"), "wb") as f:
#             f.write(sf)
#         if not args.no_code:
#             code = row.get("cadquery_file")
#             if code:
#                 try:
#                     txt = code.decode("utf-8") if isinstance(code, (bytes, bytearray)) else str(code)
#                     with open(os.path.join(args.stage_dir, uid + ".cadquery.py"), "w") as f:
#                         f.write(txt)
#                 except Exception as e:
#                     print(f"  cadquery.py write failed for {uid}: {e}")
#         staged += 1

#     print(f"Staged {staged}/{len(target_ids)} parts -> {args.stage_dir}")

#     # Provenance sidecar (same schema as select_zero_to_cad.py)
#     params = {
#         "split": args.split,
#         "source": "pinned_ids",
#         "ids_file": args.ids_file,
#         "n_requested": len(target_ids),
#         "n_staged": staged,
#         "seed": None,
#         "stratify": False,
#         "bin_edges_arg": None,
#         "bin_bounds": None,
#         "n_bins": None,
#         "min_faces": None,
#         "max_faces": None,
#     }
#     with open(os.path.join(args.stage_dir, "_select_params.json"), "w") as f:
#         json.dump(params, f, indent=2)
#     print(f"Wrote _select_params.json")


# if __name__ == "__main__":
#     main()
