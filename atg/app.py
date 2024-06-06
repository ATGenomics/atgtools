from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer

from atg.div.app import div_app
from atg.lefse.app import lefse_app
from atg.parser.app import parser_app
from atg.tools.app import tools_app
from atg.utils import OrderCommands


try:
    __version__ = version("atgtools")
except PackageNotFoundError:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"


    with open(pyproject) as f:
        toml = f.read()

    version = next(x for x in toml.splitlines() if x.startswith("version"))
    __version__ = version.split("=")[1].strip()

main_app = typer.Typer(
    help="CLI for ATGtools", cls=OrderCommands, no_args_is_help=True, add_completion=False, rich_markup_mode="rich"

)


@main_app.command("version", help="ATGtools version")
def app_version():
    typer.secho(__version__, fg=typer.colors.BRIGHT_CYAN, bold=True)



# Tools
main_app.add_typer(typer_instance=tools_app, name="tools")

# LEfSe
main_app.add_typer(typer_instance=lefse_app, name="lefse")
#
# # Parser
# main_app.add_typer(typer_instance=parser_app, name="parser")
#
# # Plots
# # main_app.add_typer(typer_instance=plot_app, name="plot")
#
# # Stats
# # main_app.add_typer(typer_instance=stats_app, name="stats")
#
# # Diversity
# main_app.add_typer(typer_instance=div_app, name="div")

