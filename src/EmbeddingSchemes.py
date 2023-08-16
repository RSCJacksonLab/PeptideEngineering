#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''

'''

import numpy as np
import pandas as pd

true_aa_name_ls = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "Dab", "Hr", "Hse", "Orn", "Bip", "Bpa"]

# integer dictionary
def get_integer_dict():
    integer_dict = {symbol: np.array([i]) for i, symbol in enumerate(true_aa_name_ls + ["-"])}
    return integer_dict

# OHE dictionary
def get_ohe_dict():
    idx_dict = {symbol: i for i, symbol in enumerate(true_aa_name_ls + ["-"])}
    arr_dict = {key: np.eye(1, len(idx_dict), k=val) for (key, val) in idx_dict.items()}
    return arr_dict

# ST Scores dictionary
def get_st_score_dict():
    st_scores = pd.read_csv("/Users/u5802006/Library/CloudStorage/GoogleDrive-dana.matthews@outlook.com.au/My Drive/2023_Venomous_peptides/GitHubRepo/PeptideEngineering/encoding_schemes/ST_scores.csv")
    aa_name_ls = [
        "Arginine", "Alanine", "Asparagine", "Aspartic acid", "Cysteine", 
        "Glutamine", "Glutamic acid", "Glycine", "Histidine", "Isoleucine", 
        "Leucine", "Lysine", "Methionine", "Phenylalanine", "Proline", "Serine", 
        "Threonine", "Tryptophan", "Tyrosine", "Valine", "2,4-Diaminobutyric acid", 
        "Homoarginine", "Homoserine", "Ornithine", "β-(4,4′-Biphenyl)alanine", 
        "4′-Benzoylphenylalanine"
        ]
    st_scores_reduced = st_scores[st_scores.Name.isin(aa_name_ls)]
    st_scores_arr = st_scores_reduced.to_numpy()[:, -8:].astype(np.float64)
    st_score_dict = {name: st_scores_arr[i] for i, name in enumerate(true_aa_name_ls)}
    st_score_dict["-"] = np.zeros((8))
    return st_score_dict


# T scores dictionary
def get_t_score_dict():
    t_scores = pd.read_csv("../encoding_schemes/T-scales.csv")
    aa_name_ls2 = ["Arginine", "Alanine", "Asparagine", "Aspartic-acid", "Cysteine", "Glutamine", "Glutamic-acid", "Glycine", "Histidine", "Isoleucine", "Leucine", "Lysine", "Methionine", "Phenylalanine", "Proline", "Serine", "Threonine", "Tryptophan", "Tyrosine", "Valine", "2,4-Diaminobutyric-acid", "Homoarginine", "Homoserine", "Ornithine", "b-(4,40-Biphenyl)alanine", "40-Benzoylphenylalanine"]
    t_scores_reduced = t_scores[t_scores.Name.isin(aa_name_ls2)]
    t_scores_arr = t_scores_reduced.to_numpy()[:, -5:].float()
    t_score_dict = {name: t_scores_arr[i] for i, name in enumerate(true_aa_name_ls)}
    t_score_dict["-"] = np.zeros((5))
    return t_score_dict


def get_embedding(seq_ls: list, emb_dict: dict):
    emb_arr = []
    for seq in seq_ls:
        seq_emb = np.array([emb_dict[token] for token in seq])
        emb_arr.append(seq_emb.flatten())
    emb_arr = np.stack(emb_arr)
    return emb_arr

### DEBUGGING 

input_tokens = [["A", "Bpa", "-"], ["A", "N", "-"]]
ohe_dict = get_st_score_dict()
encoded_tokens = get_embedding(input_tokens, ohe_dict)
print(encoded_tokens)