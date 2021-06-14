#!/usr/bin/env python

import argparse
import math
import os
import sys
from collections import defaultdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# import matplotlib.pylab

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


def read_params():
    parser = argparse.ArgumentParser(description="Plot results")
    parser.add_argument(
        "input_file", metavar="INPUT_FILE", type=str, help="tab delimited input file"
    )
    parser.add_argument(
        "output_file",
        metavar="OUTPUT_FILE",
        type=str,
        help="the file for the output image",
    )
    parser.add_argument(
        "--feature_font_size",
        dest="feature_font_size",
        type=int,
        default=7,
        help="the file for the output image",
    )
    parser.add_argument(
        "--format",
        dest="format",
        choices=["png", "svg", "pdf"],
        default="png",
        type=str,
        help="the format for the output file",
    )
    parser.add_argument("--dpi", dest="dpi", type=int, default=300)
    parser.add_argument("--title", dest="title", type=str, default="")
    parser.add_argument(
        "--title_font_size", dest="title_font_size", type=str, default="12"
    )
    parser.add_argument(
        "--class_legend_font_size",
        dest="class_legend_font_size",
        type=str,
        default="10",
    )
    parser.add_argument("--width", dest="width", type=float, default=7.0)
    parser.add_argument(
        "--height",
        dest="height",
        type=float,
        default=4.0,
        help="only for vertical histograms",
    )
    parser.add_argument("--left_space", dest="ls", type=float, default=0.2)
    parser.add_argument("--right_space", dest="rs", type=float, default=0.1)
    parser.add_argument(
        "--autoscale", dest="autoscale", type=int, choices=[0, 1], default=1
    )
    parser.add_argument(
        "--background_color",
        dest="back_color",
        type=str,
        choices=["k", "w"],
        default="w",
        help="set the color of the background",
    )
    parser.add_argument(
        "--subclades",
        dest="n_scl",
        type=int,
        default=1,
        help="number of label levels to be dislayed (starting \
                        from the leaves, -1 means all the levels, 1 is default)",
    )
    parser.add_argument(
        "--max_feature_len",
        dest="max_feature_len",
        type=int,
        default=60,
        help="Maximum length of feature strings (def 60)",
    )
    parser.add_argument("--all_feats", dest="all_feats", type=str, default="")
    parser.add_argument(
        "--otu_only",
        dest="otu_only",
        default=False,
        action="store_true",
        help="Plot only species resolved OTUs (as opposed to all levels)",
    )
    parser.add_argument(
        "--report_features",
        dest="report_features",
        default=False,
        action="store_true",
        help="Report important features to STDOUT",
    )
    args = parser.parse_args()
    return vars(args)


def read_data(input_file, output_file, otu_only):
    with open(input_file, "r") as inp:
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
    classes = list(set([v[2] for v in rows if len(v) > 2]))
    if len(classes) < 1:
        print("No differentially abundant features found in " + input_file)
        os.system("touch " + output_file)
        sys.exit()
    datar = {"rows": rows, "cls": classes}
    return datar


def plot_histo_hor(path, paramshor, datahor, bcl, report_features):
    cls2 = []
    if paramshor["all_feats"] != "":
        cls2 = sorted(paramshor["all_feats"].split(":"))
    cls = sorted(datahor["cls"])
    if bcl:
        datahor["rows"].sort(
            key=lambda ab: math.fabs(float(ab[3])) * (cls.index(ab[2]) * 2 - 1)
        )
    else:
        mmax = max([math.fabs(float(a)) for a in list(zip(*datahor["rows"]))[3]])
        datahor["rows"].sort(
            key=lambda ab: math.fabs(float(ab[3])) / mmax + (cls.index(ab[2]) + 1)
        )
    pos = np.arange(len(datahor["rows"]))
    head = 0.75
    tail = 0.5
    ht = head + tail
    ints = max(len(pos) * 0.2, 1.5)
    fig = plt.figure(
        figsize=(paramshor["width"], ints + ht),
        edgecolor=paramshor["back_color"],
        facecolor=paramshor["back_color"],
    )
    ax = fig.add_subplot(111, frame_on=False, facecolor=paramshor["back_color"])
    ls, rs = paramshor["ls"], 1.0 - paramshor["rs"]
    plt.subplots_adjust(
        left=ls,
        right=rs,
        top=1 - head * (1.0 - ints / (ints + ht)),
        bottom=tail * (1.0 - ints / (ints + ht)),
    )

    fig.canvas.set_window_title("LDA results")

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
            edgecolor=paramshor["fore_color"],
        )
    mv = max([abs(float(v[3])) for v in datahor["rows"]])
    if report_features:
        print("OTU\tLDA_score\tCLass")
        for i in out_datahor:
            print("%s\t%s\t%s" % (i, out_datahor[i][0], out_datahor[i][1]))
    for i, r in enumerate(datahor["rows"]):
        indcl = cls.index(datahor["rows"][i][2])
        if paramshor["n_scl"] < 0:
            rr = r[0]
        else:
            rr = ".".join(r[0].split(".")[-paramshor["n_scl"] :])
        if len(rr) > paramshor["max_feature_len"]:
            rr = (
                rr[: paramshor["max_feature_len"] / 2 - 2]
                + " [..]"
                + rr[-paramshor["max_feature_len"] / 2 + 2 :]
            )
        if m * (indcl * 2 - 1) < 0 and bcl:
            ax.text(
                mv / 40.0,
                float(i) - 0.3,
                rr,
                l_align,
                size=paramshor["feature_font_size"],
                color=paramshor["fore_color"],
            )
        else:
            ax.text(
                -mv / 40.0,
                float(i) - 0.3,
                rr,
                r_align,
                size=paramshor["feature_font_size"],
                color=paramshor["fore_color"],
            )
    ax.set_title(
        paramshor["title"],
        size=paramshor["title_font_size"],
        y=1.0 + head * (1.0 - ints / (ints + ht)) * 0.8,
        color=paramshor["fore_color"],
    )

    ax.set_yticks([])
    ax.set_xlabel("LDA SCORE (log 10)")
    ax.set_axisbelow(True)
    ax.xaxis.grid(linestyle="--", linewidth=0.8, dashes=(2, 3), color="gray", alpha=0.5)
    xlim = ax.get_xlim()
    if paramshor["autoscale"]:
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
        prop={"size": paramshor["class_legend_font_size"]},
    )

    def get_col_attr(x):
        return hasattr(x, "set_color") and not hasattr(x, "set_facecolor")

    for o in leg.findobj(get_col_attr):
        o.set_color(paramshor["fore_color"])
    for o in ax.findobj(get_col_attr):
        o.set_color(paramshor["fore_color"])

    plt.savefig(
        path,
        format=paramshor["format"],
        facecolor=paramshor["back_color"],
        edgecolor=paramshor["fore_color"],
        dpi=paramshor["dpi"],
    )
    plt.close()


if __name__ == "__main__":
    params = read_params()

    if "k" == params["back_color"]:
        params["fore_color"] = "w"
    else:
        params["fore_color"] = "k"
    data = read_data(params["input_file"], params["output_file"], params["otu_only"])

    plot_histo_hor(
        params["output_file"],
        params,
        data,
        len(data["cls"]) == 2,
        params["report_features"],
    )
