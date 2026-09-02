"""Argument fragments the tools share.

Nine tools each declared ``--dataset-root``, ``--split``, ``--images``,
``--device`` and ``--output``, and each embedded its own default dataset path.
Relocating the data meant editing nine files and hoping none was missed. These
functions add the same flags with the same names, help text and defaults, so a
flag means the same thing whichever tool it is given to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import default_dataset_root


def add_dataset_arguments(parser: argparse.ArgumentParser, split: str = "validation",
                          images: int | None = 16) -> argparse.ArgumentParser:
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root(),
                        help="anonymous ImageFolder root; defaults to $FRAPPE_DATASET_ROOT")
    parser.add_argument("--split", default=split)
    if images is not None:
        parser.add_argument("--images", type=int, default=images,
                            help="how many images of the split to use")
    return parser


def add_device_argument(parser: argparse.ArgumentParser,
                        default: str = "cuda:0") -> argparse.ArgumentParser:
    parser.add_argument("--device", default=default)
    return parser


def add_output_argument(parser: argparse.ArgumentParser,
                        help_text: str = "write the report here as JSON",
                        ) -> argparse.ArgumentParser:
    parser.add_argument("--output", type=Path, default=None, help=help_text)
    return parser


def resolve_device(requested: str) -> str:
    """Fall back to the CPU rather than failing when no GPU is present.

    Every tool here is runnable without a GPU -- slower, but runnable -- and a
    machine without CUDA should get results rather than a traceback.
    """
    return requested if torch.cuda.is_available() else "cpu"
