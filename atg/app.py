from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import toml
import typer
from atg.div.app import div_app
from atg.lefse.app import lefse_app
from atg.parser.app import parser_app
from atg.tools.app import tools_app

try:
    __version__ = version("atgtools")
except PackageNotFoundError:
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    __version__ = toml.load(pyproject)["tool"]["poetry"]["version"]


main_app = typer.Typer(
    help="CLI for ATGtools", add_completion=False, rich_markup_mode="rich"
)


@main_app.command("version", help="ATGtools version")
def app_version():
    typer.secho(__version__, fg=typer.colors.BRIGHT_GREEN, bold=True)


# Tools
main_app.add_typer(typer_instance=tools_app, name="tools")

# LEfSe
main_app.add_typer(typer_instance=lefse_app, name="lefse")

# Parser
main_app.add_typer(typer_instance=parser_app, name="parser")

# Plots
# main_app.add_typer(typer_instance=plot_app, name="plot")

# Stats
# main_app.add_typer(typer_instance=stats_app, name="stats")

# Diversity
main_app.add_typer(typer_instance=div_app, name="div")
