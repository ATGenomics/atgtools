#!/usr/bin/env python

import math
import pickle
import random
import argparse
import numpy as np
import rpy2.robjects as robjects


def read_params():
    parser = argparse.ArgumentParser(description='LEfSe')
    parser.add_argument('input_file', metavar='INPUT_FILE', type=str, help="the input file")
    parser.add_argument('output_file', metavar='OUTPUT_FILE', type=str,
                        help="the output file containing the data for the visualization module")
    parser.add_argument('-o', dest="out_text_file", metavar='str',
                        type=str, default="", help="set the file for exporting the result "
                                                   "(only concise textual form)")
    parser.add_argument('-a', dest="anova_alpha", metavar='float', type=float, default=0.05,
                        help="set the alpha value for the Anova test (default 0.05)")
    parser.add_argument('-w', dest="wilcoxon_alpha", metavar='float', type=float, default=0.05,
                        help="set the alpha value for the Wilcoxon test (default 0.05)")
    parser.add_argument('-l', dest="lda_abs_th", metavar='float', type=float, default=2.0,
                        help="set the threshold on the absolute value of the logarithmic LDA score"
                             "(default 2.0)")
    parser.add_argument('--nlogs', dest="nlogs", metavar='int', type=int, default=3,
                        help="max log ingluence of LDA coeff")
    parser.add_argument('-v', dest="verbose", metavar='int', choices=[0, 1], type=int,
                        default=0, help="verbose execution (default 0)")
    parser.add_argument('--wilc', dest="wilc", metavar='int', choices=[0, 1], type=int, default=1,
                        help="wheter to perform the Wicoxon step (default 1)")
    parser.add_argument('-b', dest="n_boots", metavar='int', type=int, default=30,
                        help="set the number of bootstrap iteration for LDA (default 30)")
    parser.add_argument('-e', dest="only_same_subcl", metavar='int', type=int, default=0,
                        help="set whether perform the wilcoxon test only among the subclasses "
                             "with the same name (default 0)")
    parser.add_argument('-c', dest="curv", metavar='int', type=int, default=0,
                        help="set whether perform the wilcoxon testing the Curtis's "
                             "approach [BETA VERSION] (default 0)")
    parser.add_argument('-f', dest="f_boots", metavar='float', type=float, default=0.67,
                        help="set the subsampling fraction value for each bootstrap iteration (default 0.66666)")
    parser.add_argument('-s', dest="strict", choices=[0, 1, 2], type=int, default=0,
                        help="set the multiple testing correction options. 0 no correction (more strict, "
                             "default), 1 correction for independent comparisons, 2 correction for "
                             "dependent comparison")
    parser.add_argument('--min_c', dest="min_c", metavar='int', type=int, default=10,
                        help="minimum number of samples per subclass for performing wilcoxon test (default 10)")
    parser.add_argument('-t', dest="title", metavar='str', type=str, default="",
                        help="set the title of the analysis (default input file without extension)")
    parser.add_argument('-y', dest="multiclass_strat", choices=[0, 1], type=int, default=0,
                        help="(for multiclass tasks) set whether the test is performed in a one-against-one "
                             "(1 - more strict!) or in a one-against-all setting (0 - less strict) (default 0)")

    args = parser.parse_args()

    parameters = vars(args)
    if parameters['title'] == "":
        parameters['title'] = parameters['input_file'].split("/")[-1].split('.')[0]

    return parameters


def init():
    random.seed(1986)
    robjects.r('library(splines)')
    robjects.r('library(stats4)')
    robjects.r('library(survival)')
    robjects.r('library(mvtnorm)')
    robjects.r('library(modeltools)')
    robjects.r('library(coin)')
    robjects.r('library(MASS)')


def load_data(input_file, nnorm=False):
    with open(input_file, 'rb') as inputf:
        inp = pickle.load(inputf)
    if nnorm:
        return (inp['feats'], inp['cls'],
                inp['class_sl'], inp['subclass_sl'],
                inp['class_hierarchy'], inp['norm'])
    else:
        return (inp['feats'], inp['cls'],
                inp['class_sl'], inp['subclass_sl'],
                inp['class_hierarchy'])


def get_class_means(cl_sl, feat):
    means = {}
    clk = list(cl_sl.keys())
    for fk, f in list(feat.items()):
        means[fk] = []
        for k in clk:
            means[fk].append(np.mean(f[cl_sl[k][0]:cl_sl[k][1]]))
    return clk, means


def test_kw_r(clskw, featskw, p, factors):
    robjects.globalenv["y"] = robjects.FloatVector(featskw)
    for i, f in enumerate(factors):
        vec = robjects.FactorVector(robjects.StrVector(clskw[f]))
        robjects.globalenv['x' + str(i + 1)] = vec
    fo = "y~x1"
    kw_res = robjects.r('kruskal.test(' + fo + ',)$p.value')
    return tuple(kw_res)[0] < p, tuple(kw_res)[0]


def test_rep_wilcoxon_r(sl, cl_hie, featsw, th, multiclass_strat,
                        mul_cor, fn, min_c, comp_only_same_subcl, curv=False):
    comp_all_sub = not comp_only_same_subcl
    tot_ok = 0
    alpha_mtc = th
    all_diff = []
    for pair in [(x, y) for x in list(cl_hie.keys()) for y in list(cl_hie.keys()) if x < y]:
        dir_cmp = "not_set"
        l_subcl1, l_subcl2 = [len(cl_hie[pair[0]]), len(cl_hie[pair[1]])]
        if mul_cor != 0:
            if 2 == mul_cor:
                alpha_mtc = th * l_subcl1 * l_subcl2
            else:
                alpha_mtc = 1.0 - math.pow(1.0 - th, l_subcl1 * l_subcl2)
        ok = 0
        curv_sign = 0
        first = True
        for i, k1 in enumerate(cl_hie[pair[0]]):
            br = False
            for j, k2 in enumerate(cl_hie[pair[1]]):
                if not comp_all_sub:
                    if k1[len(pair[0]):] != k2[len(pair[1]):]:
                        ok += 1
                        continue
                cl1 = featsw[sl[k1][0]:sl[k1][1]]
                cl2 = featsw[sl[k2][0]:sl[k2][1]]
                med_comp = False
                if len(cl1) < min_c or len(cl2) < min_c:
                    med_comp = True
                sx, sy = np.median(cl1), np.median(cl2)
                if cl1[0] == cl2[0] and len(set(cl1)) == 1 and len(set(cl2)) == 1:
                    tresw, first = False, False
                elif not med_comp:
                    robjects.globalenv["x"] = robjects.FloatVector(cl1 + cl2)
                    cl_li = ["a" for a in cl1] + ["b" for b in cl2]
                    vec_cl = robjects.FactorVector(robjects.StrVector(cl_li))
                    robjects.globalenv["y"] = vec_cl
                    pvw = robjects.r('pvalue(wilcox_test(x~y, data=data.frame(x, y)))')[0]
                    tresw = pvw < alpha_mtc * 2.0
                if first:
                    first = False
                    if not curv and (med_comp or tresw):
                        dir_cmp = sx < sy
                    elif curv:
                        dir_cmp = None
                        if med_comp or tresw:
                            curv_sign += 1
                            dir_cmp = sx < sy
                    else:
                        br = True
                elif not curv and med_comp:
                    if dir_cmp != (sx < sy) or sx == sy:
                        br = True
                elif curv:
                    if tresw:
                        if dir_cmp is None:
                            curv_sign += 1
                            dir_cmp = sx < sy
                    if tresw and dir_cmp != (sx < sy):
                        br = True
                        curv_sign = -1
                elif not tresw or (sx < sy) != dir_cmp or sx == sy:
                    br = True
                if br:
                    break
                ok += 1
            if br:
                break
        if curv:
            diff = curv_sign > 0
        else:
            # or (not comp_all_sub and dir_cmp != "not_set")
            diff = (ok == len(cl_hie[pair[1]]) * len(cl_hie[pair[0]]))
        if diff:
            tot_ok += 1
        if not diff and multiclass_strat:
            return False
        if diff and not multiclass_strat:
            all_diff.append(pair)
    if not multiclass_strat:
        tot_k = len(list(cl_hie.keys()))
        for k in list(cl_hie.keys()):
            nk = 0
            for a in all_diff:
                if k in a:
                    nk += 1
            if nk == tot_k - 1:
                return True
        return False
    return True


def contast_classes(featswc, inds, min_cl, ncl):
    """
    contast within classes or few per class
    """
    ff = list(zip(*[v for n, v in list(featswc.items()) if n != 'class']))
    cols = [ff[i] for i in inds]
    clswc = [featswc['class'][i] for i in inds]
    if len(set(clswc)) < ncl:
        return True
    for c in set(clswc):
        if clswc.count(c) < min_cl:
            return True
        cols_cl = [x for i, x in enumerate(cols) if clswc[i] == c]
        for i, col in enumerate(zip(*cols_cl)):
            len_01 = (len(set(col)) <= min_cl and min_cl > 1)
            len_02 = (min_cl == 1 and len(set(col)) <= 1)
            if len_01 or len_02:
                return True
    return False


def test_lda_r(clslda, featslda, cl_sl, boots,
               fract_sample, lda_th, tol_min, nlogs):
    fk = list(featslda.keys())
    means = dict([(k, []) for k in list(featslda.keys())])
    featslda['class'] = list(clslda['class'])
    clss = list(set(featslda['class']))

    for uu, k in enumerate(fk):
        if k == 'class':
            continue

        ff = [(featslda['class'][i], v) for i, v in enumerate(featslda[k])]

        for c in clss:
            max_class = max(float(featslda['class'].count(c)) * 0.5, 4)
            if len(set([float(v[1]) for v in ff if v[0] == c])) > max_class:
                continue

            for i, v in enumerate(featslda[k]):
                if featslda['class'][i] == c:
                    nor_var = random.normalvariate(0.0, max(featslda[k][i] * 0.05, 0.01))
                    featslda[k][i] = math.fabs(featslda[k][i] + nor_var)

    rdict = {}
    for a, b in list(featslda.items()):
        if a == 'class' or a == 'subclass' or a == 'subject':
            rdict[a] = robjects.StrVector(b)
        else:
            rdict[a] = robjects.FloatVector(b)

    robjects.globalenv["d"] = robjects.DataFrame(rdict)
    lfk = len(featslda[fk[0]])
    rfk = int(float(len(featslda[fk[0]])) * fract_sample)
    f = "class ~ " + fk[0]

    for k in fk[1:]:
        f += " + " + k.strip()

    ncl = len(set(clslda['class']))
    min_cl = float(min([clslda['class'].count(c) for c in set(clslda['class'])]))
    min_cl = int(min_cl * fract_sample * fract_sample * 0.5)
    min_cl = max(min_cl, 1)
    pairs = []
    for a in set(clslda['class']):
        for b in set(clslda['class']):
            if a > b:
                pairs.append((a, b))

    for k in fk:
        for i in range(boots):
            means[k].append([])

    for i in range(boots):
        for rtmp in range(1000):
            rand_s = [random.randint(0, lfk - 1) for v in range(rfk)]
            if not contast_classes(featslda, rand_s, min_cl, ncl):
                break

        rand_s = [r + 1 for r in rand_s]
        means[k][i] = []

        for p in pairs:
            robjects.globalenv["rand_s"] = robjects.IntVector(rand_s)
            robjects.globalenv["sub_d"] = robjects.r('d[rand_s,]')
            z = robjects.r('z <- suppressWarnings(lda(as.formula('
                            + f + '),data=sub_d, tol=' + str(tol_min) + '))')
            robjects.r('w <- z$scaling[,1]')
            robjects.r('w.unit <- w/sqrt(sum(w^2))')
            robjects.r('ss <- sub_d[,-match("class", colnames(sub_d))]')

            if 'subclass' in featslda:
                robjects.r('ss <- ss[,-match("subclass", colnames(ss))]')

            if 'subject' in featslda:
                robjects.r('ss <- ss[,-match("subject",colnames(ss))]')

            robjects.r('xy.matrix <- as.matrix(ss)')
            robjects.r('LD <- xy.matrix%*%w.unit')
            robjects.r('effect.size <- abs(mean(LD[sub_d[,"class"]=="'
                       + p[0] + '"]) - mean(LD[sub_d[,"class"]=="' + p[1] + '"]))')
            scal = robjects.r('wfinal <- w.unit * effect.size')
            rres = robjects.r('mm <- z$means')
            rowns = list(rres.rownames)
            lenc = len(list(rres.colnames))
            coeff = []
            for v in scal:
                if not math.isnan(float(v)):
                    coeff.append(abs(float(v)))
                else:
                    coeff.append(0.0)
            res_list = []
            for pp in [p[0], p[1]]:
                pp_v = (pp, [float(ff) for ff in rres.rx(pp, True)] if pp in rowns else [0.0] * lenc)
                res_list.append(pp_v)
            res = dict(res_list)

            for j, k in enumerate(fk):
                gm = abs(res[p[0]][j] - res[p[1]][j])
                means[k][i].append((gm + coeff[j]) * 0.5)
    res = {}
    for k in fk:
        np_max = []
        for p in range(len(pairs)):
            np_max.append(np.mean([means[k][kk][p] for kk in range(boots)]))
        m = max(np_max)
        res[k] = math.copysign(1.0, m) * math.log(1.0 + math.fabs(m), 10)
    ret_dict = dict([(k, x) for k, x in list(res.items()) if math.fabs(x) > lda_th])
    return res, ret_dict


def save_res(res, filename):
    with open(filename, 'w') as out:
        for k, v in list(res['cls_means'].items()):
            out.write(k + "\t" + str(math.log(max(max(v), 1.0), 10.0)) + "\t")
            if k in res['lda_res_th']:
                for i, vv in enumerate(v):
                    if vv == max(v):
                        out.write(str(res['cls_means_kord'][i]) + "\t")
                        break
                out.write(str(res['lda_res'][k]))
            else:
                out.write("\t")
            wc_res = "wilcox_res"
            res_wc = res[wc_res]
            out.write("\t{(res_wc[k] if wc_res in res and k in res_wc else '-')}\n")


if __name__ == '__main__':
    init()
    params = read_params()

    (feats, cls, class_sl,
     subclass_sl, class_hierarchy) = load_data(params['input_file'])

    kord, cls_means = get_class_means(class_sl, feats)
    wilcoxon_res = {}
    kw_n_ok = nf = 0

    for feat_name, feat_values in list(feats.items()):
        if params['verbose']:
            print("Testing feature", str(nf), ":", feat_name, end=' ')
            nf += 1
        kw_ok, pv = test_kw_r(cls,
                              feat_values,
                              params['anova_alpha'],
                              sorted(cls.keys()))
        if not kw_ok:
            if params['verbose']:
                print("\tkw ko")
            del feats[feat_name]
            wilcoxon_res[feat_name] = "-"
            continue

        if params['verbose']:
            print("\tkw ok\t", end=' ')

        if not params['wilc']:
            continue
        kw_n_ok += 1
        res_wilcoxon_rep = test_rep_wilcoxon_r(subclass_sl, class_hierarchy,
                                               feat_values, params['wilcoxon_alpha'],
                                               params['multiclass_strat'],
                                               params['strict'],
                                               feat_name, params['min_c'],
                                               params['only_same_subcl'],
                                               params['curv'])

        wilcoxon_res[feat_name] = str(pv) if res_wilcoxon_rep else "-"
        if not res_wilcoxon_rep:
            if params['verbose']:
                print("wilc ko")
            del feats[feat_name]
        elif params['verbose']:
            print("wilc ok\t")

    if len(feats) > 0:
        print("Number of significantly discriminative features:",
              len(feats), "(", kw_n_ok, ") before internal wilcoxon")
        k_zero = [(k, 0.0) for k, v in list(feats.items())]
        k_v = [(k, v) for k, v in list(feats.items())]
        if params['lda_abs_th'] < 0.0:
            lda_res, lda_res_th = dict(k_zero), dict()
        else:
            lda_res, lda_res_th = test_lda_r(cls, feats, class_sl,
                                             params['n_boots'],
                                             params['f_boots'],
                                             params['lda_abs_th'],
                                             0.0000000001,
                                             params['nlogs'])
    else:
        print("Number of significantly discriminative features:",
              len(feats), "(", kw_n_ok, ") before internal wilcoxon")
        print("No features with significant differences between the two classes")
        lda_res, lda_res_th = {}, {}
    outres = {'lda_res_th': lda_res_th,
              'lda_res': lda_res,
              'cls_means': cls_means,
              'cls_means_kord': kord,
              'wilcox_res': wilcoxon_res}
    print("Number of discriminative features with abs LDA score >",
          params['lda_abs_th'], ":", len(lda_res_th))

    save_res(outres, params["output_file"])
