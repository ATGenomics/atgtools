import typer

from atg.parser.kraken import kraken_output
from atg.utils import OrderCommands

parser_app = typer.Typer(
    help="Parsing tools", cls=OrderCommands, no_args_is_help=True, add_completion=False, rich_markup_mode="rich"
)



@parser_app.command(name="kraken")
def kraken_parser_command():
    """
    Parse Kraken2 output.
    """
    kraken_output()
