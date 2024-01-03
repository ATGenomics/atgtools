#!/usr/bin/env python

import math
import os
import sys
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import typer

from atg.utils import BackgroundColor, OutputFormat

app = typer.Typer()

mpl.use("Agg")
# mpl.rcParams['lines.linestyle'] = '--'

colors = [
    "#E64B35B2",
    "#00A087B2",
    "#4DBBD5B2",
    "#3C5488B2",
    "#F39B7FB2",
    "#8491B4B2",
    "#91D1C2B2",
    "#DC0000B2",
]


def read_data(input_file, output_file, otu_only):
    with open(input_file, "r", encoding="utf-8") as inp:
        if not otu_only:
            rows = []
            for line in inp.readlines():
                if len(line.strip().split()) > 3:
                    rows.append(line.strip().split()[:-1])
        else:
            # a feature with length 8 will have an OTU id associated with it
            rows = []
            for line in inp.readlines():
                if len(line.strip().split()) > 3:
                    if len(line.strip().split()[0].split(".")) == 8:
                        rows.append(line.strip().split()[:-1])
    classes = list({v[2] for v in rows if len(v) > 2})
    if len(classes) < 1:
        print(f"No differentially abundant features found in {input_file}")
        os.system("touch " + output_file)
        sys.exit()
    datar = {"rows": rows, "cls": classes}
    return datar


def plot_histo(
    path,
    pall_feats,
    pwidth,
    pback_color,
    pls,
    prs,
    pfore_color,
    pn_scl,
    pmax_feature_len,
    pfeature_font_size,
    ptitle,
    ptitle_font_size,
    pautoscale,
    pclass_legend_font_size,
    poutput_format,
    pdpi,
    datahor,
    bcl,
    report_features,
):
    cls2 = []
    if pall_feats != "":
        cls2 = sorted(pall_feats.split(":"))
    cls = sorted(datahor["cls"])
    if bcl:
        datahor["rows"].sort(
            key=lambda ab: math.fabs(float(ab[3])) * (cls.index(ab[2]) * 2 - 1)
        )
    else:
        mmax = max(math.fabs(float(a)) for a in list(zip(*datahor["rows"]))[3])
        datahor["rows"].sort(
            key=lambda ab: math.fabs(float(ab[3])) / mmax + (cls.index(ab[2]) + 1)
        )
    pos = np.arange(len(datahor["rows"]))
    head = 0.75
    tail = 0.5
    ht = head + tail
    ints = max(len(pos) * 0.2, 1.5)
    fig = plt.figure(
        figsize=(pwidth, ints + ht),
        edgecolor=pback_color,
        facecolor=pback_color,
    )
    ax = fig.add_subplot(111, frame_on=False, facecolor=pback_color)
    ls, rs = pls, 1.0 - prs
    plt.subplots_adjust(
        left=ls,
        right=rs,
        top=1 - head * (1.0 - ints / (ints + ht)),
        bottom=tail * (1.0 - ints / (ints + ht)),
    )

    # fig.canvas.manager.set_window_title("LDA results")
    fig.suptitle("LDA results")

    l_align = {"horizontalalignment": "left", "verticalalignment": "baseline"}
    r_align = {"horizontalalignment": "right", "verticalalignment": "baseline"}
    added = []
    if datahor["rows"][0][2] == cls[0]:
        m = 1
    else:
        m = -1
    out_datahor = defaultdict(list)
    for i, v in enumerate(datahor["rows"]):
        if report_features:
            otu = v[0].split(".")[7].replace("_", ".")
            score = v[3]
            otu_class = v[2]
            out_datahor[otu] = [score, otu_class]
        indcl = cls.index(v[2])
        if str(v[2]) not in added:
            lab = str(v[2])
        else:
            lab = None
        added.append(str(v[2]))
        col = colors[indcl % len(colors)]
        if len(cls2) > 0:
            col = colors[cls2.index(v[2]) % len(colors)]
        if bcl:
            vv = math.fabs(float(v[3])) * (m * (indcl * 2 - 1))
        else:
            vv = math.fabs(float(v[3]))
        ax.barh(
            pos[i],
            vv,
            align="center",
            color=col,
            label=lab,
            height=0.8,
            edgecolor=pfore_color,
        )
    mv = max(abs(float(v[3])) for v in datahor["rows"])
    if report_features:
        print("OTU\tLDA_score\tCLass")
        for i in out_datahor:
            print(f"{i}\t{out_datahor[i][0]}\t{out_datahor[i][1]}")
    for i, r in enumerate(datahor["rows"]):
        indcl = cls.index(datahor["rows"][i][2])
        if pn_scl < 0:
            rr = r[0]
        else:
            rr = ".".join(r[0].split(".")[-pn_scl:])
        if len(rr) > pmax_feature_len:
            param_max_feature_len_minus = rr[: pmax_feature_len / 2 - 2]
            param_max_feature_len_plus = rr[-pmax_feature_len / 2 + 2 :]
            rr = param_max_feature_len_minus + " [..]" + param_max_feature_len_plus
        if m * (indcl * 2 - 1) < 0 and bcl:
            ax.text(
                mv / 40.0,
                float(i) - 0.3,
                rr,
                l_align,
                size=pfeature_font_size,
                color=pfore_color,
            )
        else:
            ax.text(
                -mv / 40.0,
                float(i) - 0.3,
                rr,
                r_align,
                size=pfeature_font_size,
                color=pfore_color,
            )
    ax.set_title(
        ptitle,
        size=ptitle_font_size,
        y=1.0 + head * (1.0 - ints / (ints + ht)) * 0.8,
        color=pfore_color,
    )

    ax.set_yticks([])
    ax.set_xlabel("LDA SCORE (log 10)")
    ax.set_axisbelow(True)
    ax.xaxis.grid(linestyle="--", linewidth=0.8, dashes=(2, 3), color="gray", alpha=0.5)
    xlim = ax.get_xlim()
    if pautoscale:
        round_1 = round((abs(xlim[0]) + abs(xlim[1])) / 10, 4)
        round_2 = round(round_1 * 100, 0)
        ran = np.arange(0.0001, round_2 / 100)
        if 1 < len(ran):
            if len(ran) < 100:
                min_ax = min(xlim[1] + 0.0001, round_2 / 100)
                ax.set_xticks(np.arange(xlim[0], xlim[1] + 0.0001, min_ax))
    ax.set_ylim((pos[0] - 1, pos[-1] + 1))
    leg = ax.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=5,
        borderaxespad=0.0,
        frameon=False,
        prop={"size": pclass_legend_font_size},
    )

    def get_col_attr(x):
        return hasattr(x, "set_color") and not hasattr(x, "set_facecolor")

    for o in leg.findobj(get_col_attr):
        o.set_color(pfore_color)
    for o in ax.findobj(get_col_attr):
        o.set_color(pfore_color)

    plt.savefig(
        path,
        format=poutput_format,
        facecolor=pback_color,
        edgecolor=pfore_color,
        dpi=pdpi,
    )
    plt.close()


@app.command()
def plot_results(
    input_file: str,
    output_file: str,
    feature_font_size: int,
    output_format: OutputFormat,
    dpi: int,
    title: str,
    title_font_size: int,
    class_legend_font_size: int,
    width: int,
    left_space: float,
    right_space: float,
    autoscale: bool,
    back_color: BackgroundColor,
    n_scl: bool,
    max_feature_len: int,
    all_feats: str,
    otu_only: bool,
    report_features: bool,
):
    if "k" == back_color:
        fore_color = "w"
    else:
        fore_color = "k"
    data = read_data(input_file, output_file, otu_only)

    plot_histo(
        output_file,
        all_feats,
        width,
        back_color,
        left_space,
        right_space,
        fore_color,
        n_scl,
        max_feature_len,
        feature_font_size,
        title,
        title_font_size,
        autoscale,
        class_legend_font_size,
        output_format,
        dpi,
        data,
        len(data["cls"]) == 2,
        report_features,
    )
