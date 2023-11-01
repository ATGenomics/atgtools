from functools import wraps
from pathlib import Path
import toml
import time
from typing import Any
from pathlib import Path
import re
import shutil


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
    Convert multiline FASTA to single line FASTA
    """

    filepath = Path(input_fasta).resolve()
    shutil.copy(filepath, f"{filepath}.bak")

    output_file = input_fasta

    with open(input_fasta, "r") as fasta_file:
        fasta_data = fasta_file.read()
        sequences = re.findall(">[^>]+", fasta_data)

    with open(output_file, "w") as fasta:
        for i in sequences:
            header, seq = i.split("\n", 1)
            header += "\n"
            seq = seq.replace("\n", "") + "\n"
            fasta.write(header + seq)
