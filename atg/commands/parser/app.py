from atg.commands.parser.kraken import kraken_output
import typer


parser_app = typer.Typer(help="Miscellaneous tools.")


@parser_app.command(name="kraken")
def kraken_parser_command():
    """
    Parse Kraken2 output.
    """
    kraken_output()
