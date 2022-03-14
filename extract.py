#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pathlib
import sys #tmp debug
import time

import wandb
import torch
from tqdm import tqdm

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["mean", "per_tok", "bos", "contacts"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate sequences longer than 1024 to match the training setup",
    )
    parser.add_argument(
        "--drop_invalid",
        action="store_true",
        help="Instead of truncating sequences longer than 1024, log the error and drop them",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    parser.add_argument(
        "--torch_hub",
        type=str,
        help="Location of torch hub (if not set via environment variable). Make sure to use full path ",
    )
    return parser


def main(args):
    run = wandb.init()
    if args.torch_hub is not None:
        torch.hub.set_dir("/users/looerk/torch_hub/")
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    print("debug model loaded")
    
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            if args.truncate:
                toks = toks[:, :1022]

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            # logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(tqdm(labels)):
                idx, label = label.split("|")  # The fasta batched dataset already strips the > character
                args.output_file = args.output_dir / f"{idx}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {}  # "index": idx, "label": label
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].clone()  # Crop off the starting <s> token and the padding at the end
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone()

                torch.save(
                    result,
                    args.output_file,
                )
    # run.finish()


if __name__ == "__main__":
    print("debug start extract.py")
    start = time.perf_counter()
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    end = time.perf_counter()
    print(f"Time taken: {end-start:.3f}s")
    print("debug end")
