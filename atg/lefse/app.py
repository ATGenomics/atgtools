import typer

from atg.lefse.format import format_input
from atg.lefse.lefse import run_lefse
from atg.lefse.plot import plot_results
from atg.utils import (
    BackgroundColor,
    CorrectionLevel,
    FeaturesDir,
    OrderCommands,
    OutputFormat,
)

lefse_app = typer.Typer(help="LEfSe implementation", cls=OrderCommands)


@lefse_app.command(name="format")
def format_lefse_command(
    input_file: str = typer.Option(
        ...,
        "--input",
        "-i",
        show_default=False,
        help="the input file, feature hierarchical level "
        "can be specified with | or . and those symbols "
        "must not be present for other reasons in the "
        "input file.",
    ),
    output_file: str = typer.Option(
        ...,
        "--output",
        "-o",
        show_default=False,
        help="the output pickle file containing the data for LEfSe",
    ),
    feats_dir: FeaturesDir = typer.Option(
        "r",
        "--features",
        "-f",
        case_sensitive=False,
        show_default=True,
        help="set whether the features are on rows ('r') or on columns ('c')",
    ),
    pclass: int = typer.Option(
        1,
        "--class",
        "-c",
        show_default=True,
        help="set which feature use as class (default 1)",
    ),
    psubclass: int = typer.Option(
        None,
        "--subclass",
        "-s",
        show_default=True,
        help="set which feature use as subclass (default -1 meaning no subclass)",
    ),
    psubject: int = typer.Option(
        None,
        "--subject",
        "-u",
        show_default=True,
        help="set which feature use as subject (default -1 meaning no subject)",
    ),
    norm_v: float = typer.Option(
        -1.0,
        "--norm",
        "-n",
        show_default=True,
        help="set the normalization value (default -1.0 meaning no normalization)",
    ),
    json_format: bool = typer.Option(
        False,
        "--json",
        "-j",
        show_default=False,
        help="the formatted table in json format",
    ),
):
    """
    Format the input file for LEfSe
    """
    format_input(
        input_file,
        output_file,
        feats_dir,
        pclass,
        psubclass,
        psubject,
        norm_v,
        json_format,
    )


@lefse_app.command(name="run")
def run_lefse_command(
    input_file: str = typer.Option(
        ..., "--input", "-i", show_default=False, help="the pickle input file"
    ),
    output_file: str = typer.Option(
        ...,
        "--output",
        "-o",
        show_default=False,
        help="the output file containing the data for the visualization module",
    ),
    anova_alpha: float = typer.Option(
        0.05,
        "--anova_alpha",
        "-a",
        show_default=True,
        help="set the alpha value for the Anova test",
    ),
    wilcoxon_alpha: float = typer.Option(
        0.05,
        "--wilcoxon_alpha",
        "-w",
        show_default=True,
        help="set the alpha value for the Wilcoxon test",
    ),
    lda_abs_th: float = typer.Option(
        2.0,
        "--lda_abs_th",
        "-l",
        show_default=True,
        help="set the threshold on the absolute value of the logarithmic LDA score",
    ),
    nlogs: int = typer.Option(
        3, "--nlogs", "-n", show_default=True, help="max log influence of LDA coeff"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", show_default=True, help="verbose execution"
    ),
    wilc: bool = typer.Option(
        True,
        "--wilc",
        "-c",
        show_default=True,
        help="wheter to perform the Wicoxon step",
    ),
    n_boots: int = typer.Option(
        30,
        "--n_boots",
        "-b",
        show_default=True,
        help="set the number of bootstrap iteration for LDA",
    ),
    only_same_subcl: bool = typer.Option(
        False,
        "--only_same_subcl",
        "-e",
        show_default=True,
        help="set whether perform the wilcoxon test only "
        "among the subclasses with the same name",
    ),
    curv: bool = typer.Option(
        False,
        "--curv",
        "-r",
        show_default=True,
        help="set whether perform the wilcoxon testing the Curtis's approach "
        "[BETA VERSION] (default 0)",
    ),
    f_boots: float = typer.Option(
        0.67,
        "--f_boots",
        "-f",
        show_default=True,
        help="set the subsampling fraction value for each bootstrap "
        "iteration (default 0.66666)",
    ),
    strict: CorrectionLevel = typer.Option(
        "0",
        "--strict",
        "-s",
        show_default=True,
        help="set the multiple testing correction options. "
        "0 no correction (more strict default), "
        "1 correction for independent comparisons, "
        "2 correction for dependent comparison",
    ),
    min_c: int = typer.Option(
        10,
        "--min_c",
        "-m",
        show_default=True,
        help="minimum number of samples per subclass for " "performing wilcoxon test",
    ),
    title: str = typer.Option(
        "",
        "--title",
        "-t",
        show_default=True,
        help="set the title of the analysis (default input file without extension)",
    ),
    multiclass_strat: bool = typer.Option(
        False,
        "--multiclass_strat",
        "-y",
        show_default=True,
        help="(for multiclass tasks) set whether the test is performed in "
        "a one-against-one (1 - more strict!) or in a one-against-all "
        "setting (0 - less strict) (default 0)",
    ),
):
    """
    Run LEfSe
    """
    run_lefse(
        input_file,
        output_file,
        anova_alpha,
        wilcoxon_alpha,
        lda_abs_th,
        nlogs,
        verbose,
        wilc,
        n_boots,
        only_same_subcl,
        curv,
        f_boots,
        strict,
        min_c,
        title,
        multiclass_strat,
    )


@lefse_app.command(name="plot")
def plot_lefse_command(
    input_file: str = typer.Option(
        ..., "--input", "-i", show_default=False, help="tab delimited input file"
    ),
    output_file: str = typer.Option(
        ..., "--output", "-o", show_default=False, help="the file for the output image"
    ),
    feature_font_size: int = typer.Option(
        7,
        "--feature-font-size",
        "-f",
        show_default=True,
        help="the font size for the features",
    ),
    output_format: OutputFormat = typer.Option(
        "png",
        "--format",
        "-t",
        show_default=True,
        help="the format for the output image",
    ),
    dpi: int = typer.Option(
        300, "--dpi", show_default=True, help="the dpi for the output image"
    ),
    title: str = typer.Option(
        "", "--title", "-t", show_default=False, help="the title for the plot"
    ),
    title_font_size: int = typer.Option(
        12,
        "--title-font-size",
        "-tf",
        show_default=True,
        help="the font size for the title",
    ),
    class_legend_font_size: int = typer.Option(
        10,
        "--class-legend-font-size",
        "-cf",
        show_default=True,
        help="the font size for the class legend",
    ),
    width: int = typer.Option(
        7, "--width", "-w", show_default=True, help="the width of the plot"
    ),
    left_space: float = typer.Option(
        0.2, "--left-space", "-ls", show_default=True, help="the left space of the plot"
    ),
    right_space: float = typer.Option(
        0.1,
        "--right-space",
        "-rs",
        show_default=True,
        help="the right space of the plot",
    ),
    autoscale: bool = typer.Option(
        True, "--autoscale", "-a", show_default=True, help="autoscale the plot"
    ),
    back_color: BackgroundColor = typer.Option(
        "w", "--background-color", "-bc", show_default=True, help="the background color"
    ),
    n_scl: bool = typer.Option(
        False,
        "--subclades",
        "-s",
        show_default=True,
        help="number of label levels to be dislayed (starting \
                        from the leaves, -1 means all the levels, 1 is default)",
    ),
    max_feature_len: int = typer.Option(
        60,
        "--max-feature-len",
        "-m",
        show_default=True,
        help="the maximum length of the feature name",
    ),
    all_feats: str = typer.Option(
        "", "--all-feats", "-af", show_default=False, help="show all features"
    ),
    otu_only: bool = typer.Option(
        False,
        "--otu-only",
        "-o",
        show_default=True,
        help="Plot only species resolved OTUs (as opposed to all levels)",
    ),
    report_features: bool = typer.Option(
        False,
        "--report-features",
        "-rf",
        show_default=False,
        help="report features to STDOUT",
    ),
):
    plot_results(
        input_file,
        output_file,
        feature_font_size,
        output_format,
        dpi,
        title,
        title_font_size,
        class_legend_font_size,
        width,
        left_space,
        right_space,
        autoscale,
        back_color,
        n_scl,
        max_feature_len,
        all_feats,
        otu_only,
        report_features,
    )
