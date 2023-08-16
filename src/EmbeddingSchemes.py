#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''

'''

#import string
import numpy as np
import pandas as pd

true_aa_name_ls = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "Dab", "Hr", "Hse", "Orn", "Bip", "Bpa"]

def get_ohe_dict(additional_tokens: list=None):
    if additional_tokens is None:
        additional_tokens = []

    idx_dict = {symbol: i for i, symbol in enumerate(true_aa_name_ls + additional_tokens)}
    arr_dict = {key: np.eye(1, len(idx_dict), k=val) for (key, val) in idx_dict.items()}
    return arr_dict

#additional tokens
#termini_symbols = ["*", "#"]
gap = ["-"]
additional_tokens = gap
ohe_dict = get_ohe_dict(additional_tokens)

# Bridget to functions that return dictionaries for any other embedding schemes
#
# i.e. in the form {"A": np.array([1, 2, 3, 4], "B": np.array([1, 2, 3, 4]))}
#       where [1, 2, 3, 4] are values of the vector that represent that amino acid.



# ST Scores dictionary

def get_st_score_dict(add_token_ls: list = None):
    if not add_token_ls:
        add_token_ls = []
    st_scores = pd.read_csv("../encoding_schemes/ST_scores.csv")
    aa_name_ls = ["Arginine", "Alanine", "Asparagine", "Aspartic acid", "Cysteine", "Glutamine", "Glutamic acid", "Glycine", "Histidine", "Isoleucine", "Leucine", "Lysine", "Methionine", "Phenylalanine", "Proline", "Serine", "Threonine", "Tryptophan", "Tyrosine", "Valine", "2,4-Diaminobutyric acid", "Homoarginine", "Homoserine", "Ornithine", "β-(4,4′-Biphenyl)alanine", "4′-Benzoylphenylalanine"]
    st_scores_reduced = st_scores[st_scores.Name.isin(aa_name_ls)]
    st_scores_arr = st_scores_reduced.to_numpy()[:, -8:]
    st_score_dict = {name: st_scores_arr[i] for i, name in enumerate(true_aa_name_ls)}
    st_score_dict["-"] = np.zeros((8))
    # st_score_dict["#"] =
    # st_score_dict["*"] =
    return st_score_dict


# T scores dictionary

def get_t_score_dict(add_token_ls: list = None):
    if not add_token_ls:
        add_token_ls = []
    t_scores = pd.read_csv("../encoding_schemes/T-scales.csv")
    aa_name_ls2 = ["Arginine", "Alanine", "Asparagine", "Aspartic-acid", "Cysteine", "Glutamine", "Glutamic-acid", "Glycine", "Histidine", "Isoleucine", "Leucine", "Lysine", "Methionine", "Phenylalanine", "Proline", "Serine", "Threonine", "Tryptophan", "Tyrosine", "Valine", "2,4-Diaminobutyric-acid", "Homoarginine", "Homoserine", "Ornithine", "b-(4,40-Biphenyl)alanine", "40-Benzoylphenylalanine"]
    t_scores_reduced = t_scores[t_scores.Name.isin(aa_name_ls2)]
    t_scores_arr = t_scores_reduced.to_numpy()[:, -5:]
    t_score_dict = {name: t_scores_arr[i] for i, name in enumerate(true_aa_name_ls)}
    t_score_dict["-"] = np.zeros((5))
    # t_score_dict["#"] = 
    # t_score_dict["*"] =
    return t_score_dict

# integer dictionary (complete)

def get_integer_dict(additional_tokens: list = None):
    if additional_tokens is None:
        additional_tokens = []
    integer_dict = {symbol: i for i, symbol in enumerate(true_aa_name_ls + additional_tokens)}
    return integer_dict
# Additional tokens
#termini_symbols = ["*", "#"]
gap = ["-"]
additional_tokens = gap

integer_dict = get_integer_dict(additional_tokens)
