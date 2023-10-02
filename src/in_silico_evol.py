from argparse import ArgumentParser
from Bio import SeqIO
import EmbeddingSchemes
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle as pkl
from sklearn.utils import shuffle
from tqdm import tqdm

emb_name_dict = {
    "int": EmbeddingSchemes.get_integer_dict(),
    "ohe": EmbeddingSchemes.get_ohe_dict(),
    "st-score": EmbeddingSchemes.get_st_score_dict(),
    "t-score": EmbeddingSchemes.get_t_score_dict()
}

def load_mut_dict(mut_dict_path: Path, assume_python_indexing: bool=False):
    '''
    Load a dictionary containing possible mutations for each site from a
    JSON file.
    '''
    mut_dict = json.load(open(mut_dict_path))
    # if indexing is not pythonic
    if not assume_python_indexing:
        mut_dict = {key-1: mut_dict[key] for key in mut_dict.keys()}
    return mut_dict

def mutate_sequences(seq_ls: list, name_ls: list, mut_dict: dict):
    '''
    Given a dictionary containing possible mutations, make every possible
    mutation on a list of sequences.

    Assumes sequences contain amino acids separated by "_".
    '''
    seq_arr = np.array([seq.split("_") for seq in seq_ls])
    mut_gap_seq_ls = []
    mut_name_ls = []
    mut_seq_ls = []
    # for each sequence
    for i in tqdm(range(seq_arr.shape[0])):
        seq = seq_arr[i, :]
        # for each site in the sequence
        for j in range(seq.shape[0]):
            site_aa = seq[j]
            # skip mutation site if insertion
            if site_aa == "-":
                continue
            possible_aa_ls = mut_dict[j]
            # make each possible mutation
            for aa in possible_aa_ls:
                mut_seq = seq.copy()
                # skip mutation if results in the same sequence
                if mut_seq[j] == aa:
                    continue
                # make mutation
                mut_seq[j] = aa
                mut_seq_str = "".join(list(mut_seq))
                mut_name = name_ls[i] + f"_{j+1}{aa}"
                if not mut_seq_str in mut_seq_ls:
                    mut_gap_seq_ls.append(list(mut_seq))
                    mut_seq_ls.append(mut_seq_str)
                    mut_name_ls.append(mut_name)
    return mut_gap_seq_ls, mut_name_ls

def in_silico_step(
        current_seq_ls: list, 
        current_name_ls: list, 
        mut_dict: dict, 
        sklearn_model_dict: dict
    ):
    '''
    Single round of in silico evolution. Takes already mutated sequences, 
    makes predictions, determines specificity ratio score, prunes sequences
    and then makes mutations for the next round.
    '''
    # results for this round
    res_dict = {
        "names": current_name_ls, 
        "sequences": ["_".join(s) for s in current_seq_ls]
    }

    for model_name in tqdm(sklearn_model_dict, desc="Models"):
        
        # initialise loaded model
        model = sklearn_model_dict[model_name]
        _, embedding_name, _ = model_name.split("_")

        # embed sequences
        emb_dict = emb_name_dict[embedding_name]
        emb_seqs = EmbeddingSchemes.get_embedding(current_seq_ls, emb_dict)

        # get 1.7 predictions
        # add 1 to embeddings for 1.7
        emb_seqs_7 = np.concatenate(
            (emb_seqs, np.ones((emb_seqs.shape[0], 1))*1),
            axis=1
        )
        pred_iq50_7 = model.predict(emb_seqs_7)
        res_dict[f"{model_name}_pred_nav1.7_IC50"] = list(pred_iq50_7)

        # get 1.1 predictions
        # add 2 to embeddings for 1.1
        emb_seqs_1 = np.concatenate(
            (emb_seqs, np.ones((emb_seqs.shape[0], 1))*2), 
            axis=1
        )
        pred_iq50_1 = model.predict(emb_seqs_1)
        res_dict[f"{model_name}_pred_nav1.1_IC50"] = list(pred_iq50_1)

        # get 1.6 predictions
        # add 3 to embeddings for 1.6
        emb_seqs_6 = np.concatenate(
            (emb_seqs, np.ones((emb_seqs.shape[0], 1))*3),
            axis=1
        )
        pred_iq50_6 = model.predict(emb_seqs_6)
        res_dict[f"{model_name}_pred_nav1.6_IC50"] = list(pred_iq50_6)

        # get ratio
        pred_ratio = (pred_iq50_1 + pred_iq50_7)/(2*pred_iq50_6)
        res_dict[f"{model_name}_pred_ratio"] = pred_ratio

        # get normalised ratio 
        min_iq50_1 = np.min(pred_iq50_1)
        range_iq50_1 = np.max(pred_iq50_1) - min_iq50_1
        norm_iq50_1 = (pred_iq50_1 - min_iq50_1)/range_iq50_1 + 1e-4

        min_iq50_7 = np.min(pred_iq50_7)
        range_iq50_7 = np.max(pred_iq50_7) - min_iq50_7
        norm_iq50_7 = (pred_iq50_7 - min_iq50_7)/range_iq50_7 + 1e-4

        min_iq50_6 = np.min(pred_iq50_6)
        range_iq50_6 = np.max(pred_iq50_6) - min_iq50_6
        norm_iq50_6 = (pred_iq50_6 - min_iq50_6)/range_iq50_6 + 1e-4

        pred_norm_ratio = (norm_iq50_1 + norm_iq50_7)/(2*norm_iq50_6)
        res_dict[f"{model_name}_pred_ratio_normalised"] = pred_norm_ratio

    for channel in ["nav1.7_IC50", "nav1.1_IC50", "nav1.6_IC50"]:
        # get predictions for each channel
        ratio_dict = {
            key: res_dict[key] for key in res_dict.keys()
            if f"pred_{channel}" in key
        }
        pred_arr = pd.DataFrame(ratio_dict).to_numpy()
        mean_preds = np.mean(pred_arr, axis=1)
        stdev_preds = np.std(pred_arr, axis=1)
        res_dict[f"mean_{channel}_predictions"] = mean_preds
        res_dict[f"stdev_{channel}_predictions"] = stdev_preds

    # get average and std of prediction
    in_silico_ratio_dict = {
        key: res_dict[key] for key in res_dict.keys()
        if "pred_ratio_normalised" in key
    }
    pred_arr = pd.DataFrame(in_silico_ratio_dict).to_numpy()

    # average of ratio predictions
    mean_preds = np.mean(pred_arr, axis=1)
    res_dict["mean_norm_ratio_predictions"] = mean_preds
    
    # standard deviation of ratio predictions
    stdev_preds = np.std(pred_arr, axis=1)
    res_dict["stdev_norm_ratio_predictions"] = stdev_preds

    # 95% upper confidence bound of ratio predictions
    ucb95_preds = mean_preds + 1.96*(stdev_preds)
    res_dict["ucb_norm_ratio_predictions"] = ucb95_preds

    # prune sequences
    res_df = pd.DataFrame(res_dict)

    # remove duplicates
    res_df.drop_duplicates(subset="sequences")
    
    # take top 200 ratios (lowest predicted ratio score)
    top_preds = res_df.sort_values(
        "mean_norm_ratio_predictions", 
        ascending=True
    )[:200]
    remaining_seqs = res_df[200:]
    # take top 200 upper confidence bounds
    top_ucbs = remaining_seqs.sort_values(
        "ucb_norm_ratio_predictions",
        ascending=True
    )[:200]
    remaining_seqs = res_df[200:]
    shuffle(remaining_seqs)
    # take a random 200 peptides
    random_seqs = remaining_seqs[:200]
    pruned_seq_df = pd.concat([top_preds, top_ucbs, random_seqs])

    # mutate sequences for the next round
    next_seqs, next_names = mutate_sequences(
        pruned_seq_df.sequences.tolist(), 
        pruned_seq_df.names.tolist(), 
        mut_dict
    )    
    return next_seqs, next_names, pruned_seq_df, res_df

def create_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--n', 
        type=Path, 
        required=True,
        description="Number of in silico rounds",
    )
    parser.add_argument(
        '--model_dir', 
        type=Path, 
        required=True,
        description="Path to directory containing pickled sklearn models",
    )
    parser.add_argument(
        '--start_seqs',
        type=Path,
        required=True,
        descrption="Path to fasta containing starting sequences",
    )
    parser.add_argument(
        '--mut_dict',
        type=Path,
        required=True,
        description="Path to json containing allowed mutations.",
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        description="Path to save outputs.",
    )
    return parser

if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()
    n_rounds = args.n
    model_dir = args.model_dir
    start_seqs = args.start_seqs
    mut_dict_path = args.mut_dict
    output_dir = args.output

    # parse starting data
    seq_parser = SeqIO.parse(start_seqs, "fasta")
    seqs = [str(seq.seq).split("_") for seq in seq_parser]
    names = [str(seq.id).upper().rstrip() for seq in seq_parser]

    # load models for ensemble
    loaded_models = {}
    for model in model_dir:
        model_name = model[:-4]
        model = pkl.load(open(f"./results/final_models/{model}", "rb"))
        loaded_models[model_name] = model

    # get mutation dictionary 
    mut_dict = load_mut_dict(mut_dict_path)
    
    # in silico evolution
    for round in range(n_rounds): 
        seqs, names, new_res_dict, all_res_dict = in_silico_step(
            seqs, 
            names, 
            mut_dict, 
            loaded_models
        )

        # save results to res_dict
        pruned_path = os.path.join(
            output_dir, 
            f"in_silico_PRUNED_round{round}.csv"
        )
        pd.DataFrame(new_res_dict).to_csv(pruned_path, index=False)
        all_seqs_path = os.path.join(
            output_dir, 
            f"in_silico_ALL_round{round}.csv"
        )
        pd.DataFrame(all_res_dict).to_csv(all_seqs_path, index=False)