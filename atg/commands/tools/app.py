import typer
from atg.commands.tools.manifest import create_manifest
from atg.utils import one_liner
# from atg.commands.config.delete import delete_config
# from atg.commands.config.describe import describe_config
# from atg.commands.config.list import list_configs
# from atg.commands.etc.help_text import CONFIG_ARG_HELP

tools_app = typer.Typer(help="Miscellaneous tools.")


ROOT_COMMAND_HELP = """\b
Directory containing FASTQ files.
[green bold]Supported name scheme:[/green bold]\b
[green]+ sample_R1.fastq.gz[green]
[green]+ sample_R2.fastq.gz[green]
[green]+ sample_S01_R1.fastq.gz[green]
[green]+ sample_S01_R2.fastq.gz[green]
[red bold]Not supported name scheme:[/red bold]\b 
[red]- sample_S01_L001_R1_001.fastq.gz[/red]
[red]- sample_S01_L001_R2_001.fastq.gz[/red]
"""


@tools_app.command(name="manifest")
def manifest_tools_command(
        fastq_dir: str = typer.Option(...,
                                      "--fastq_dir",
                                      "-d",
                                      show_default=False,
                                      help= ROOT_COMMAND_HELP),
        output_file: str = typer.Option("manifest.tsv",
                                        "--output",
                                        "-o",
                                        show_default=False,
                                        help="Output file name. [default: manifest.tsv]"),
        csv_format: bool = typer.Option(False, "--csv", "-c",
                                        help="Output CSV file format"),
):
    """
    Create the manifest.[tsv/csv] file for QIIME2. [default: tsv]
    """

    create_manifest(fastq_dir=fastq_dir, output_file=output_file, csv_format=csv_format)


@tools_app.command(name="oneliner")
def oneliner_tools_command(
        input_fasta_file: str = typer.Argument(show_default=False,
                                               help="Input fasta file")):
    """
    Convert multiline FASTA to single line FASTA
    """
    one_liner(input_fasta_file)
