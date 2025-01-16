import typer

from atg.plots.plots import venn_diagram, volcano
from atg.utils import OrderCommands

plot_app = typer.Typer(
    help="Common plots for diversity analysis",
    cls=OrderCommands,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


@plot_app.command(name="venn", help="Plot Venn diagram of an abundance tables")
def venn_command():
    venn_diagram()


@plot_app.command(name="volcano", help="Plot Venn diagram of an abundance tables")
def volcano_command():
    volcano()
