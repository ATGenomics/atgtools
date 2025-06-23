import typer
from pathlib import Path
from typing import Optional

from atg.div.alpha import AlphaDiversity
from atg.div.beta import BetaDiversity
from atg.utils import OrderCommands, get_abundance

div_app = typer.Typer(
    help="Alpha and Beta diversity",
    cls=OrderCommands,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


@div_app.command(name="alpha", help="Calculate Alpha diversity Indexes")
def alpha_div_command(
    input_file: Optional[Path] = typer.Option(None, help="Path to abundance data file"),
    num_equiv: bool = typer.Option(False, help="Calculate number equivalents")
):
    """Calculate alpha diversity indices for the given abundance data."""
    abundance_df = get_abundance(input_file)
    alpha_div = AlphaDiversity(abundance_df)
    results = alpha_div.calculate_diversity_indices(num_equiv=num_equiv)
    print(results)


@div_app.command(name="beta", help="Calculate dissimilarity between samples")
def beta_div_command(
    input_file: Optional[Path] = typer.Option(None, help="Path to abundance data file"),
    metric: str = typer.Option("braycurtis", help="Distance metric to use")
):
    """Calculate beta diversity (dissimilarity) between samples."""
    abundance_df = get_abundance(input_file)
    beta_div = BetaDiversity(abundance_df)
    distances = beta_div.calculate_pairwise_distances(metric)
    print(distances)
