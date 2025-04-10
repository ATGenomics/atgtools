import re
import shutil
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any

import anndata as ad
import pyfastx
from click import Context
from loguru import logger
from typer.core import TyperGroup


def timeit(f: Any) -> Any:
    """
    Calculate the time it takes to run a function
    """

    @wraps(f)
    def wrapper(*args, **kargs):  # type: ignore
        start = time.time()
        result = f(*args, **kargs)
        end = time.time()
        res = round((end - start), 4)
        logger.info(f"Elapsed time {f.__name__}: {res} secs")
        return result

    return wrapper


def one_liner(input_fasta: str) -> None:
    """
    Convert multiline FASTA to single line FASTA. The input file is overwritten.
    """

    filepath = Path(input_fasta).resolve()
    shutil.copy(filepath, f"{filepath}.bak")

    output_file = input_fasta

    with open(input_fasta, "r", encoding="utf-8") as fasta_file:
        fasta_data = fasta_file.read()
        sequences = re.findall(">[^>]+", fasta_data)

    with open(output_file, "w", encoding="utf-8") as fasta:
        for i in sequences:
            header, seq = i.split("\n", 1)
            header += "\n"
            seq = seq.replace("\n", "") + "\n"
            fasta.write(header + seq)


class FeaturesDir(str, Enum):
    ROWS = "r"
    COLS = "c"


class CorrectionLevel(str, Enum):
    NO_CORRECTION = 0
    INDEPENDENT_COMP = 1
    DEPENDENT_COMP = 2


class OutputFormat(str, Enum):
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"


class BackgroundColor(str, Enum):
    WHITE = "w"
    BLACK = "k"


class OrderCommands(TyperGroup):
    def list_commands(self, _ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)


def get_abundance():
    se = ad.read_h5ad(Path(__file__).parents[1] / "tests/zeller14.h5ad")
    se.var_names = se.var["taxonomy"].str.replace(r"^.+([a-z]__.+$)", "\\1", regex=True)
    # df = se.to_df().T.rename_axis("features").reset_index()
    return se.to_df().T.reset_index()


def check_dir(fastq_dir: str) -> Path:
    """
    Check if a directory exists and is not empty.

    Args:
        fastq_dir: Path to the directory to check

    Returns:
        Path: Resolved path to the directory
    """
    _fastq_dir = Path(fastq_dir).resolve()
    if not any(Path(_fastq_dir).iterdir()):
        raise FileNotFoundError(f"{_fastq_dir.stem}/ is empty")
    return _fastq_dir


def fastq_files(fastq: str, pattern: str) -> list:
    if Path(fastq).is_file():
        fqfile = Path(fastq).name.partition(".")[0]
        return {fqfile: fastq}

    check_dir(fastq)
    pattern = re.compile(r".*_([1-2]|R[1-2]).(fastq|fq)\.gz$")
    fqfiles = sorted([x for x in Path(fastq).glob("*") if pattern.match(str(x))])
    snames = sorted([str(x.name.partition(".")[0]) for x in fqfiles])

    return dict(zip(snames, fqfiles))


def count_fastq(fastq_file, pattern: str):
    _fastq_files = fastq_files(fastq=fastq_file, pattern=pattern)
    for k, v in _fastq_files.items():
        print(k, len(pyfastx.Fastx(str(v), build_index=True)))
        index_file = Path(f"{str(v)}.fxi")
        if index_file.exists():
            index_file.unlink()
