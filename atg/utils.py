import re
import shutil
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any

from click import Context
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
        print(f"Elapsed time {f.__name__}: {res} secs", end="\n")
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
    rows = "r"
    cols = "c"


class CorrectionLevel(str, Enum):
    no_correction = 0
    independent_comp = 1
    dependent_comp = 2


class OutputFormat(str, Enum):
    png = "png"
    svg = "svg"
    pdf = "pdf"


class BackgroundColor(str, Enum):
    white = "w"
    black = "k"


class OrderCommands(TyperGroup):
    def list_commands(self, ctx: Context):
        """Return list of commands in the order appear."""
        return list(self.commands)
