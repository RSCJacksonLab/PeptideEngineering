import numpy as np

updated_mut_dict = {
    1: ['G', '-'], 
    2: ['N', 'E', '-', 'G', 'D'], 
    3: ['D', 'E', 'A'], 
    4: ['C'], 
    5: ['L', 'K'], 
    6: ['G', 'E', 'Q'], 
    7: ['F', 'Z', 'X', 'W', 'P', 'M'], 
    8: ['W', 'F', 'G'], 
    9: ['S', 'O', 'K', 'E'], 
    10: ['A', 'R', 'S', 'G', 'K'], 
    11: ['C'], 
    12: ['N', 'S', 'V', 'D'], 
    13: ['P'], 
    14: ['K', 'R', 'G', 'N', 'S'], 
    15: ['N', 'K'], 
    16: ['D', 'N'], 
    17: ['K', 'R', 'E'], 
    18: ['C'], 
    19: ['C'], 
    20: ['A', 'P', 'S', 'E', 'K'], 
    21: ['-', 'S'], 
    22: ['-', 'S'], 
    23: ['N', 'G', 'S'], 
    24: ['L', 'Y', 'R'], 
    25: ['V', 'K', 'T'], 
    26: ['C'], 
    27: ['S', 'H', 'R'], 
    28: ['S', 'K', 'N', 'W', 'Q'], 
    29: ['K', 'R'], 
    30: ['H', 'D', 'Y', 'L'], 
    31: ['K', 'B', 'J', 'R', 'U', 'L', 'P', 'Q'], 
    32: ['W', 'Z', 'X', 'T'], 
    33: ['C'], 
    34: ['K', 'J', 'R', 'U'], 
    35: ['G', 'A', 'Y', 'V'], 
    36: ['K', 'U', 'R', 'L', 'D', 'A', 'G'], 
    37: ['L', 'I', 'W'], 
    38: ['-', 'W']
}



# Import mutation dictionary and sequence dataset. using updated_mut_dict as above
input_sequences = np.load('/Users/bridget/Desktop/Honours/code stuff/PeptideEngineering/data/fastas/test_arr.npy')

input_sequences = array([['Hs1a_K14S', 'G', 'N', 'D', 'C', 'L', 'G', 'F', 'W', 'S', 'A',
        'C', 'N', 'P', 'S', 'N', 'D', 'K', 'C', 'C', 'A', '-', '-', 'N',
        'L', 'V', 'C', 'S', 'S', 'K', 'H', 'K', 'W', 'C', 'K', 'G', 'K',
        'L', '-'],
       ['Hs1a_H28D', 'G', 'N', 'D', 'C', 'L', 'G', 'F', 'W', 'S', 'A',
        'C', 'N', 'P', 'K', 'N', 'D', 'K', 'C', 'C', 'A', '-', '-', 'N',
        'L', 'V', 'C', 'S', 'S', 'K', 'D', 'K', 'W', 'C', 'K', 'G', 'K',
        'L', '-'],
       ['Hs1a_MPNN', 'G', 'N', 'D', 'C', 'L', 'Q', 'P', 'G', 'E', 'R',
        'C', 'N', 'P', 'K', 'N', 'D', 'K', 'C', 'C', 'A', '-', '-', 'N',
        'L', 'V', 'C', 'S', 'K', 'R', 'L', 'R', 'T', 'C', 'K', 'A', 'A',
        'L', '-']], dtype='<U9')


def mutate_sequences(sequences, names, mut_dict):
    mutated_sequences = []
    mutated_names = []
    for i, sequence in enumerate(sequences):
        sequence_name = names[i]
        current_sequence = sequence
        for position in mut_dict.keys():
            for mutations in mut_dict[position]:
                if current_sequence[position] != mutations:
                    mutated_sequence = current_sequence.copy()
                    mutated_sequence[position] = mutations
                    new_name = sequence_name + f"+{current_sequence[position]}{position+1}{mutations}"
                    mutated_names.append(new_name)
                    mutated_sequences.append(mutated_sequence)
    return mutated_sequences, mutated_names

num_rounds = 2
current_sequences = input_sequences[1:, :].copy()
current_names = input_sequences[0,:].copy()

for round in range(num_rounds):
    print(f"Round {round+1}")
    current_sequences, current_names = mutate_sequences(current_sequences, current_names, mut_dict)
    print(f"Number of sequences after mutation: {len(current_sequences)}")
    print(f"Number of names after mutation: {len(current_names)}")
    for i, sequence in enumerate(current_sequences):
        print(f"Mutated Name {i + 1}: {current_names[i]}")
        print(f"Mutated Sequence {i + 1}: {sequence}")