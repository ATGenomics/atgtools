import typer

from atg.div.alpha import alpha_diversity
from atg.div.beta import beta_diversity
from atg.utils import OrderCommands, get_abundance

div_app = typer.Typer(
    help="Alpha and Beta diversity",
    cls=OrderCommands,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


@div_app.command(name="alpha", help="Calculate Alpha diversity Indexes")
def alpha_div_command():
    print(alpha_diversity(input_df=get_abundance(), num_equiv=False))


@div_app.command(name="beta", help="Calculate disimilarity between samples")
def beta_div_command():
    beta_diversity()
