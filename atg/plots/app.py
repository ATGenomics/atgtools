import typer
from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing_extensions import Annotated

from atg.plots.venn import VennPlotter, VennConfig
from atg.plots.volcano import VolcanoPlotter, VolcanoConfig
from atg.plots.upset import UpSetPlotter, UpSetConfig
from atg.utils import OrderCommands

plot_app = typer.Typer(
    help="Common plots for diversity analysis",
    cls=OrderCommands,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


@plot_app.command(name="venn", help="Plot Venn diagram of three sets")
def venn_command(
    input_file: Path = typer.Argument(..., help="Path to input CSV file with three columns"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the plot (PNG/SVG)"),
    colors: List[str] = typer.Option(None, "--colors", "-c", help="Colors for the three sets (comma-separated)"),
    alpha: float = typer.Option(0.75, "--alpha", "-a", help="Transparency of the circles"),
    figsize: Annotated[Tuple[int, int], typer.Option(help="Figure size (width, height)")] = (8, 8),
    set_labels: Annotated[Tuple[str, str, str], typer.Option(help="Labels for the three sets")] = ("Set A", "Set B", "Set C"),
):
    """Create a Venn diagram from three sets of data.
    
    The input CSV file should contain three columns, each representing a set.
    The plot will show the intersections between these sets.
    """
    # Create config
    config = VennConfig(
        set_labels=set_labels,
        colors=colors,
        alpha=alpha,
        figsize=figsize
    )
    
    # Create plotter
    plotter = VennPlotter(config)
    
    # Create plot
    fig = plotter.plot(input_file, save=output_file)
    
    # Show if no output file
    if not output_file:
        plt.show()


@plot_app.command(name="volcano", help="Create a volcano plot from differential expression results")
def volcano_command(
    input_file: Path = typer.Argument(..., help="Path to input CSV file with DE results"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the plot (PNG/SVG)"),
    log2fc: str = typer.Option("log2FoldChange", "--log2fc", help="Column name for log2 fold change"),
    pvalue: str = typer.Option("padj", "--pvalue", help="Column name for p-values"),
    symbol: str = typer.Option("symbol", "--symbol", help="Column name for gene symbols"),
    base_mean: Optional[str] = typer.Option(None, "--base-mean", help="Column name for base mean values"),
    pval_thresh: float = typer.Option(0.05, "--pval-thresh", help="P-value threshold"),
    log2fc_thresh: float = typer.Option(0.75, "--log2fc-thresh", help="Log2 fold change threshold"),
    to_label: int = typer.Option(5, "--to-label", help="Number of top genes to label"),
    colors: List[str] = typer.Option(None, "--colors", help="Colors for different categories (comma-separated)"),
    figsize: Annotated[Tuple[int, int], typer.Option(help="Figure size (width, height)")] = (5, 5),
):
    """Create a volcano plot from differential expression results.
    
    The input CSV file should contain columns for log2 fold change, p-values,
    and gene symbols. The plot will show differentially expressed genes.
    """
    # Create config
    config = VolcanoConfig(
        log2fc=log2fc,
        pvalue=pvalue,
        symbol=symbol,
        base_mean=base_mean,
        pval_thresh=pval_thresh,
        log2fc_thresh=log2fc_thresh,
        to_label=to_label,
        colors=colors,
        figsize=figsize
    )
    
    # Create plotter
    plotter = VolcanoPlotter(config)
    
    # Create plot
    fig = plotter.plot(input_file, save=output_file)
    
    # Show if no output file
    if not output_file:
        plt.show()


@plot_app.command(name="upset", help="Create an UpSet plot from boolean columns in a CSV file")
def upset_command(
    input_file: Path = typer.Argument(..., help="Path to input CSV file with boolean columns"),
    val_col: Optional[str] = typer.Option(None, "--value-column", "-v", help="Column name for weighted counts"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save the plot (PNG/SVG)"),
    categories_colors: Optional[List[str]] = typer.Option(None, "--colors", "-c", help="Colors for categories (comma-separated)"),
    categorylabel_size: int = typer.Option(20, "--label-size", "-l", help="Size of category labels"),
    categorylabel_color: str = typer.Option("black", "--label-color", help="Color of category labels"),
    datalabel_color: str = typer.Option("black", "--data-color", help="Color of data labels"),
    datalabel_size: int = typer.Option(20, "--data-size", help="Size of data labels"),
    marker_size: int = typer.Option(20, "--marker-size", help="Size of intersection markers"),
    markerline_color: str = typer.Option("black", "--marker-color", help="Color of marker lines"),
    bar_intersect_color: str = typer.Option("black", "--bar-color", help="Color of intersection bars"),
    figsize: Annotated[Tuple[int, int], typer.Option(help="Figure size (width, height)")] = (12, 8),
):
    """Create an UpSet plot from boolean columns in a CSV file.
    
    The input CSV file should contain boolean columns representing different categories.
    The plot will show the intersections between these categories.
    """
    # Create config
    config = UpSetConfig(
        categories_colors=categories_colors,
        categorylabel_size=categorylabel_size,
        categorylabel_color=categorylabel_color,
        datalabel_color=datalabel_color,
        datalabel_size=datalabel_size,
        marker_size=marker_size,
        markerline_color=markerline_color,
        bar_intersect_color=bar_intersect_color,
        figsize=figsize
    )
    
    # Create plotter
    plotter = UpSetPlotter(config)
    
    # Create plot
    fig = plotter.plot(input_file, val_col)
    
    # Save or show
    if output_file:
        fig.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()
