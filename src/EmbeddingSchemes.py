#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''

'''

import string
import numpy as np


def get_ohe_dict(add_token_ls: list=None):
    '''
    Returns a dictionary with amino acids as keys and the corresponding
    one-hot vector as values.

    add_token_ls : list, default=None
        Optional list of additional amino acids to return a ohe vector
        for. For example noncanonical amino acids, gap tokens, termini
        symbols.
    '''
    if not add_token_ls:
        add_token_ls = []
    can_aa_ls = [aa for aa in string.ascii_uppercase if aa not in "BJOUXZ"]
    ## need to add check that no canonical aas have been provided as additional tokense
    idx_dict = {symbol: i for i, symbol in enumerate(can_aa_ls + add_token_ls)}
    arr_dict = {key: np.eye(1, len(idx_dict), k=val) for (key, val) in idx_dict.items()}
    return arr_dict

# Bridget to functions that return dictionaries for any other embedding schemes
#
# i.e. in the form {"A": np.array([1, 2, 3, 4], "B": np.array([1, 2, 3, 4]))}
#       where [1, 2, 3, 4] are values of the vector that represent that amino acid.





### example
unnatural_aas = ["Bpa", "Bip", "Orn", "Hse", "Hr", "Dab"]
termini_symbols = ["*", "#"]
gap = ["-"]
additional_tokens = unnatural_aas + termini_symbols + gap

ohe_dict = get_ohe_dict(add_token_ls=additional_tokens)