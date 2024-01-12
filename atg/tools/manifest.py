import sys
from pathlib import Path
from typing import Dict

from atg.utils import check_dir, timeit


@timeit
def create_manifest(fastq_dir: str, output_file: str, csv_format: bool) -> None:
    """
    Create a manifest file (tsv/csv) from a directory containing FASTQ files.
    """

    _fastq_dir = check_dir(fastq_dir)

    output_manifest: Dict[str, str] = {}
    output = Path.cwd() / output_file
    fq_files = [str(x.name) for x in _fastq_dir.glob("*fastq.gz")]
    prefix = sorted({"_".join(i.split("_")[:-1]) for i in fq_files})

    def table_format(manifest: dict, file: Path, comma: bool = False) -> None:
        if comma:
            sep = ","
            out = Path(str(file).split(".", maxsplit=1)[0] + ".csv")
        else:
            sep = "\t"
            out = output

        for sample in prefix:
            manifest[sample] = (
                f"{_fastq_dir}/{sample}_R1.fastq.gz{sep}"
                f"{_fastq_dir}/{sample}_R2.fastq.gz"
            )

        if out.is_file():
            print("There is a previous manifest file")
            sys.exit()
        else:
            with open(out, "w", encoding="utf-8") as f:
                headers = [
                    "sample-id",
                    "forward-absolute-filepath",
                    "reverse-absolute-filepath",
                ]
                f.write(f"{sep}".join(headers) + "\n")

                for k, v in manifest.items():
                    f.write(f"{k}{sep}{v}" + "\n")

    table_format(output_manifest, output, csv_format)
