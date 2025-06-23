import re
import shutil
import time
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Union, Dict, Callable, List

import anndata as ad
import pyfastx
from click import Context
from loguru import logger
from typer.core import TyperGroup
from tqdm.contrib.concurrent import thread_map


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


def thread_map_parallel(func: Callable, data: List, max_workers: int) -> None:
    """Execute function in parallel using thread_map.
    
    Args:
        func: Function to execute
        data: List of data items to process
        max_workers: Maximum number of worker threads
    """
    if len(data) > 0:
        thread_map(func, data, max_workers=max_workers, desc="Processing")


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


def fastq_files(fastq: str, pattern: str) -> dict:
    """
    Get FASTQ files from directory or single file.
    
    Args:
        fastq: Path to FASTQ file or directory
        pattern: Regex pattern to match files
        
    Returns:
        dict: Dictionary mapping sample names to file paths
    """
    if Path(fastq).is_file():
        fqfile = Path(fastq).name.partition(".")[0]
        return {fqfile: fastq}

    check_dir(fastq)
    pattern = re.compile(pattern)
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


def check_paired(fastq_files: dict) -> dict:
    """
    Check if FASTQ files are paired-end based on naming patterns.
    
    Args:
        fastq_files: Dictionary of FASTQ files from fastq_files function
        
    Returns:
        dict: Dictionary with 'paired' and 'single' keys containing file lists
    """
    paired_files = {}
    single_files = {}
    
    # Common paired-end patterns (both compressed and uncompressed)
    paired_patterns = [
        (r'_R1\.(fastq|fq)(\.gz)?$', r'_R2\.(fastq|fq)(\.gz)?$'),
        (r'_1\.(fastq|fq)(\.gz)?$', r'_2\.(fastq|fq)(\.gz)?$'),
        (r'\.1\.(fastq|fq)(\.gz)?$', r'\.2\.(fastq|fq)(\.gz)?$'),
    ]
    
    for sample_name, file_path in fastq_files.items():
        file_str = str(file_path)
        is_paired = False
        
        for pattern1, pattern2 in paired_patterns:
            if re.search(pattern1, file_str):
                # Find corresponding R2/2 file
                potential_pair = re.sub(pattern1, pattern2, file_str)
                if Path(potential_pair).exists():
                    paired_files[sample_name] = {
                        'R1': file_path,
                        'R2': Path(potential_pair)
                    }
                    is_paired = True
                    break
            elif re.search(pattern2, file_str):
                # Find corresponding R1/1 file
                potential_pair = re.sub(pattern2, pattern1, file_str)
                if Path(potential_pair).exists():
                    paired_files[sample_name] = {
                        'R1': Path(potential_pair),
                        'R2': file_path
                    }
                    is_paired = True
                    break
        
        if not is_paired:
            single_files[sample_name] = file_path
    
    return {'paired': paired_files, 'single': single_files}
