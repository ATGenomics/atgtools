import typer

from atg.parser.kraken import kraken_output

parser_app = typer.Typer(help="Parsing tools")


@parser_app.command(name="kraken")
def kraken_parser_command():
    """
    Parse Kraken2 output.
    """
    kraken_output()
