#!/usr/bin/env python

import json
import pickle
import re
from enum import Enum
from functools import cmp_to_key

import numpy as np
import typer

app = typer.Typer()


def read_input_file(inp_file):
    with open(inp_file) as inp:
        data = [
            [v.strip() for v in line.strip().split("\t")] for line in inp.readlines()
        ]
        return data


def modify_feature_names(ret):
    ascii_chars = [
        [" ", "$", "@", "#", "%", "^", "&", "*", "'"],
        ["/", "(", ")", "-", "+", "=", "{", "}", "[",
         "]", ",", ".", ";", ":", "?", "<", ">", ".", ","]]

    for p in ascii_chars[0]:
        ret = [re.sub(re.escape(p), "", f) for f in ret]

    for g in ascii_chars[1]:
        ret = [re.sub(re.escape(g), "_", f) for f in ret]

    ret = [re.sub(r"\|", ".", f) for f in ret]

    ret2 = []
    for r in ret:
        if r[0] in [str(x) for x in range(10)] + list("_"):
            ret2.append("f_" + r)
        else:
            ret2.append(r)

    return ret2


def sort_by_cl(datals, n, c, s, u):
    def sort_lines1(a, b):
        return int(a[c] > b[c]) * 2 - 1

    def sort_lines2u(a, b):
        if a[c] != b[c]:
            return int(a[c] > b[c]) * 2 - 1

        return int(a[u] > b[u]) * 2 - 1

    def sort_lines2s(a, b):
        if a[c] != b[c]:
            return int(a[c] > b[c]) * 2 - 1

        return int(a[s] > b[s]) * 2 - 1

    def sort_lines3(a, b):
        if a[c] != b[c]:
            return int(a[c] > b[c]) * 2 - 1
        if a[s] != b[s]:
            return int(a[s] > b[s]) * 2 - 1

        return int(a[u] > b[u]) * 2 - 1

    if n == 3:
        datals.sort(key=cmp_to_key(sort_lines3))

    if n == 2:
        if s is None:
            datals.sort(key=cmp_to_key(sort_lines2u))
        else:
            datals.sort(key=cmp_to_key(sort_lines2s))

    if n == 1:
        datals.sort(key=cmp_to_key(sort_lines1))

    return datals


def rename_same_subcl(cla, subcl):
    toc = []
    for sc in set(subcl):
        subclis = []
        for i in range(len(subcl)):
            if sc == subcl[i]:
                subclis.append(cla[i])
        if len(set(subclis)) > 1:
            toc.append(sc)
    new_subcl = []
    for i, sc in enumerate(subcl):
        if sc in toc:
            new_subcl.append(cla[i] + "_" + sc)
        else:
            new_subcl.append(sc)
    return new_subcl


def get_class_slices(datasl):
    prev_class = list(datasl)[-1][0]
    prev_subclass = list(datasl)[-1][1]
    subcl_slices, cl_slices, class_hrchy, subcls = ([], [], [], [])
    last_cl = last_subcl = 0

    i = None
    for i, d in enumerate(datasl):
        if prev_subclass != d[1]:
            subcl_slices.append((prev_subclass, (last_subcl, i)))
            last_subcl = i
            subcls.append(prev_subclass)
        if prev_class != d[0]:
            cl_slices.append((prev_class, (last_cl, i)))
            class_hrchy.append((prev_class, subcls))
            subcls = []
            last_cl = i
        prev_subclass = d[1]
        prev_class = d[0]
    subcl_slices.append([prev_subclass, (last_subcl, i + 1)])
    subcls.append(prev_subclass)
    cl_slices.append([prev_class, (last_cl, i + 1)])
    class_hrchy.append((prev_class, subcls))
    return dict(cl_slices), dict(subcl_slices), dict(class_hrchy)


def add_missing_levels(ff):
    if sum([f.count(".") for f in ff]) < 1:
        return ff

    clades2leaves = {}
    for f in ff:
        fs = f.split(".")
        if len(fs) < 2:
            continue
        for g in range(len(fs)):
            n = ".".join(fs[:g])
            if n in clades2leaves:
                clades2leaves[n].append(f)
            else:
                clades2leaves[n] = [f]
    for k, h in list(clades2leaves.items()):
        if k and k not in ff:
            fnvv = [[float(fn) for fn in ff[vv]] for vv in h]
            ff[k] = [sum(a) for a in zip(*fnvv)]
    return ff


def numerical_values(feat, nnorm):
    for k, va in list(feat.items()):
        feat[k] = [val for val in va]
    if nnorm <= 0:
        return feat
    tr = list(zip(*feat.values()))

    mul = []
    fk = list(feat.keys())
    if len(fk) < sum([k.count(".") for k in fk]):
        hie = True
    else:
        hie = False

    for n, p in enumerate(list(feat.values())[0]):
        if hie:
            to_sum = []
            for j, t in enumerate(tr[n]):
                if fk[j].count(".") < 1:
                    to_sum.append(float(t))
            res_sum = sum(to_sum)
            mul.append(res_sum)
        else:
            mul.append(sum(tr[n]))

    if hie and sum(mul) == 0:
        mul = []
        for i in range(len(list(feat.values())[0])):
            mul.append(sum(tr[i]))
    for i, m in enumerate(mul):
        if m == 0:
            mul[i] = 0.0
        else:
            mul[i] = nnorm / m

    for k, l in list(feat.items()):
        feat[k] = []
        for i, val in enumerate(l):
            feat[k].append(float(val) * mul[i])
        cv = np.std(feat[k]) / np.mean(feat[k])
        if np.mean(feat[k]) and cv < 1e-10:
            feat[k] = []
            for kv in feat[k]:
                num = float(round(kv * 1e6) / 1e6)
                feat[k].append(num)
    return feat


class FeaturesDir(str, Enum):
    rows = "r"
    cols = "c"


@app.command()
def format_input(
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
    data = read_input_file(input_file)
    # Transpose the data if the features are on columns
    if feats_dir == "c":
        data = list(zip(*data))

    first_line = modify_feature_names(list(zip(*data))[0])

    ncl = 1
    class_1 = pclass - 1
    if psubclass is not None:
        ncl += 1
        subclass_1 = psubclass - 1
    else:
        subclass_1 = None

    if psubject is not None:
        ncl += 1
        subject_1 = psubject - 1
    else:
        subject_1 = None

    data = list(
        zip(
            first_line,
            *sort_by_cl(list(zip(*data))[1:], ncl, class_1, subclass_1, subject_1)
        )
    )

    cls_i = [("class", pclass - 1)]
    if psubclass is not None:
        cls_i.append(("subclass", psubclass - 1))
    if psubject is not None:
        cls_i.append(("subject", psubject - 1))

    cls = {}
    for v in cls_i:
        cls[v[0]] = data[:3].pop(v[1])[1:]

    if psubclass is None:
        cls["subclass"] = []
        for cl in cls["class"]:
            cls["subclass"].append(str(cl) + "_subcl")

    cls["subclass"] = rename_same_subcl(cls["class"], cls["subclass"])

    class_sl, subclass_sl, class_hierarchy = get_class_slices(list(zip(*cls.values())))

    if psubject is not None:
        feats = dict([(d[0], d[1:]) for d in data[3:]])
    elif psubclass is not None:
        feats = dict([(d[0], d[1:]) for d in data[2:]])
    else:
        feats = dict([(d[0], d[1:]) for d in data[1:]])

    feats = add_missing_levels(feats)
    feats = numerical_values(feats, norm_v)

    out = {
        "feats": feats,
        "norm": norm_v,
        "cls": cls,
        "class_sl": class_sl,
        "subclass_sl": subclass_sl,
        "class_hierarchy": class_hierarchy,
    }

    if json_format:
        with open(output_file, "w") as back_file:
            back_file.write(
                json.dumps(out, sort_keys=True, indent=4, ensure_ascii=False)
            )
    else:
        with open(output_file, "wb") as back_file:
            pickle.dump(out, back_file)


@app.command()
def run_lefse(
        input_file: str = typer.Option(
            ..., "--input", "-i", show_default=False, help="the pickle input file"
        ), ):
    print("second command")


if __name__ == "__main__":
    app()
