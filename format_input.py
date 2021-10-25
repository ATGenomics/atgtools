#!/usr/bin/env python

import sys
import argparse
import re
import numpy as np
import json
import pickle
from functools import cmp_to_key


def read_params():
    parser = argparse.ArgumentParser(description="LEfSe formatting modules")
    parser.add_argument(
        "input_file",
        metavar="INPUT_FILE",
        type=str,
        help="the input file, feature hierarchical level can be specified "
        "with | or . and those symbols must not be present for other "
        "reasons in the input file.",
    )
    parser.add_argument(
        "output_file",
        metavar="OUTPUT_FILE",
        type=str,
        help="the output file containing the data for LEfSe",
    )
    parser.add_argument(
        "-j",
        dest="json_format",
        required=False,
        action="store_true",
        help="the formatted table in json format",
    )
    parser.add_argument(
        "-f",
        dest="feats_dir",
        choices=["c", "r"],
        type=str,
        default="r",
        help="set whether the features are on rows (default) or on columns",
    )
    parser.add_argument(
        "-c",
        dest="class",
        metavar="[1..n_feats]",
        type=int,
        default=1,
        help="set which feature use as class (default 1)",
    )
    parser.add_argument(
        "-s",
        dest="subclass",
        metavar="[1..n_feats]",
        type=int,
        default=None,
        help="set which feature use as subclass (default -1 meaning no subclass)",
    )
    parser.add_argument(
        "-u",
        dest="subject",
        metavar="[1..n_feats]",
        type=int,
        default=None,
        help="set which feature use as subject (default -1 meaning no subject)",
    )
    parser.add_argument(
        "-o",
        dest="norm_v",
        metavar="float",
        type=float,
        default=-1.0,
        help="set the normalization value (default -1.0 meaning no normalization)",
    )
    parser.add_argument(
        "-m",
        dest="missing_p",
        choices=["f", "s"],
        type=str,
        default="d",
        help="set the policy to adopt with missing values: f removes the features "
        "with missing values, s removes samples with missing values (default f)",
    )
    args = parser.parse_args()
    return vars(args)


def read_input_file(inp_file):
    common = {"ReturnedData": []}

    with open(inp_file) as inp:
        for line in inp.readlines():
            li = []
            for r in line.strip().split("\t"):
                li.append(r.strip())
            common["ReturnedData"].append(li)
    return common


def transpose(table):
    return list(zip(*table))


def modify_feature_names(fn):
    ret = fn
    ascii = [
        [" ", "$", "@", "#", "%", "^", "&", "*", "'"],
        ["/", "(", ")", "-", "+", "=", "{", "}", "[",
         "]", ",", ".", ";", ":", "?", "<", ">", ".",
         ",", ],
    ]

    for p in ascii[0]:
        ret = [re.sub(re.escape(p), "", f) for f in ret]

    for g in ascii[1]:
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


if __name__ == "__main__":
    params = read_params()
    # print(params)
    common_area = read_input_file(sys.argv[1])
    # print(common_area)
    # print(list(common_area.items())[0][1][-1])
    # print(list(common_area.items())[0][1][-1][0:2])

    data = common_area["ReturnedData"]
    # print(data)

    if params["feats_dir"] == "c":
        data = transpose(data)
        # print(data)

    first_line = list(zip(*data))[0]
    # print(first_line[-1])
    first_line = modify_feature_names(list(first_line))
    # print(first_line[4:])

    ncl = 1
    class_1 = params["class"] - 1
    if params["subclass"] is not None:
        ncl += 1
        subclass_1 = params["subclass"] - 1
    else:
        subclass_1 = None
    if params["subject"] is not None:
        ncl += 1
        subject_1 = params["subject"] - 1
    else:
        subject_1 = None

    # subclass_1 = params['subclass'] - 1 if params['subclass'] is not None else None
    # subject_1 = params['subject'] - 1 if params['subject'] is not None else None

    data = list(
        zip(
            first_line,
            *sort_by_cl(list(zip(*data))[1:], ncl, class_1, subclass_1, subject_1)
        )
    )

    # for x in [x[0:] for x in data[0:4]]:
    #    print('\t'.join([str(i) for i in x]))

    cls_i = [("class", params["class"] - 1)]
    if params["subclass"] is not None:
        cls_i.append(("subclass", params["subclass"] - 1))
    if params["subject"] is not None:
        cls_i.append(("subject", params["subject"] - 1))

    # cls_i.sort(lambda x, y: -cmp(x[1], y[1]))
    # print(cls_i)

    # print(data[2])
    cls = {}
    # cls_i.sort(key=lambda x: (-x[1]))
    for v in cls_i:
        cls[v[0]] = data[:3].pop(v[1])[1:]
    # print(data[0:3])
    # print(cls)

    # print({k:v for (k, v) in cls.items()})

    # python 2 code: if not params['subclass'] > 0
    # print(cls.items())
    # print(params.items())
    if params["subclass"] is None:
        cls["subclass"] = []
        for cl in cls["class"]:
            cls["subclass"].append(str(cl) + "_subcl")
    # print(cls.items())
    # print(cls)

    cls["subclass"] = rename_same_subcl(cls["class"], cls["subclass"])

    class_sl, subclass_sl, class_hierarchy = get_class_slices(list(zip(*cls.values())))
    # print(data)

    if params["subject"] is not None:
        feats = dict([(d[0], d[1:]) for d in data[3:]])
    elif params["subclass"] is not None:
        feats = dict([(d[0], d[1:]) for d in data[2:]])
    else:
        feats = dict([(d[0], d[1:]) for d in data[1:]])

    feats = add_missing_levels(feats)
    norm = params["norm_v"]
    feats = numerical_values(feats, norm)

    out = {
        "feats": feats,
        "norm": norm,
        "cls": cls,
        "class_sl": class_sl,
        "subclass_sl": subclass_sl,
        "class_hierarchy": class_hierarchy,
    }

    if params["json_format"]:
        with open(params["output_file"], "w") as back_file:
            back_file.write(
                json.dumps(out, sort_keys=True, indent=4, ensure_ascii=False)
            )
    else:
        with open(params["output_file"], "wb") as back_file:
            pickle.dump(out, back_file)
