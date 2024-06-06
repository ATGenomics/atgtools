import numpy as np
import pandas as pd


def alpha_div():
    pass


def shannon(y):
    """
    Shannon's H natural logarith in base e.
    H = -sum( p_i * log(p_i) )
    where p_i is relative abundance of species i
    """
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    t = t[t != 0]
    H = -np.sum(t * np.log(t))
    return H


def shannon2(y, base=2):
    """
    Shannon's H natural logarithm in base 2.
    H = -sum( p_i * log(p_i) )
    """
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    t = t[t != 0]
    H = -(np.sum(t * np.log(t))) / np.log(base)
    return H


def gini(y):
    """
    Gini-Simpson
    D = 1 - sum( p_i^2 )
    """
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    D = 1 - np.sum(t**2)
    return D


def simpson(y):
    """
    Simpson's D
    D = sum(p_i^2)
    """
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    D = np.sum(t**2)
    return D


def dom(y):
    """
    Dominance index
    D = max(p_i)
    """
    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    D = np.max(t)
    return D


def richness(y):
    """
    Species richness
    D = Number of non-zero observations
    """
    notabs = ~np.isnan(y)
    t = y[notabs]
    D = np.sum(t != 0)
    return float(D)


def evenness(y):
    """
    Evenness: Shannon's H divided by the log of species richness
    """
    return shannon(y) / richness(y)


def pielou_e(y):
    """
    Pielou's J
    J = H / ln(S)
    where H is the Shannon-Wiener entropy and S the richness.
    """
    J = shannon(y) / np.log(richness(y))
    return J


def alpha_diversity(input_df, level=None, num_equiv=True):
    """
    Computes a given diversity index for a site x species matrix.
    All indices computed along rows (axis = 1)

    Parameters
    ----------
    df:  numpy array or pandas dataframe with observations as rows
        and descriptors as columns
    num_equiv: Whether or not the diversity index return species number equivalents,
        i.e. the number of species of identical abundance. This has better properties
        than raw diversity. The number equivalents are as follows:

        shannon: np.exp(H)
        gini-simpson: 1 / (1-D)
        simpson: 1/D
    """

    if level is None:
        tmp = input_df.loc[:, "taxonomy":].set_index("taxonomy").T
    else:
        tmp = input_df.groupby(level).sum().T

    tmpid = tmp.index

    indexmethods = {
        "Shannon": shannon,
        "Shannon2": shannon2,
        "Gini-Simpson": gini,
        "Simpson": simpson,
        "Dominance": dom,
        "Richness": richness,
        "Pielou": pielou_e,
    }
    z: None = None
    if not isinstance(tmp, pd.DataFrame):
        msg = "x argument must be a pandas dataframe"
        raise ValueError(msg)
    if isinstance(tmp, pd.DataFrame):
        if (tmp.dtypes == "object").any():
            msg = "DataFrame can only contain numeric values"
            raise ValueError(msg)
        if (tmp < 0).any().any():
            msg = "DataFrame contains negative values"
            raise ValueError(msg)
        z = np.array(tmp, "float")

    z = z / z.sum(axis=1)[:, np.newaxis]

    results = []
    for k, v in indexmethods.items():
        div = np.apply_along_axis(v, 1, z)
        if num_equiv:
            if k == "Shannon":
                div = np.exp(div)
            if k == "Shannon2":
                div = np.exp(div)
            if k == "Gini-Simpson":
                div = 1.0 / (1.0 - div)
            if k == "Simpson":
                div = 1.0 / div

            tmp_df = pd.DataFrame(div, index=tmpid, columns=[k])
            results.append(tmp_df)
        else:
            tmp_df = pd.DataFrame(div, index=tmpid, columns=[k])
            results.append(tmp_df)

    div_df = pd.concat(results, axis=1)
    return div_df


# ic(alpha_diversity(df, num_equiv=False))
