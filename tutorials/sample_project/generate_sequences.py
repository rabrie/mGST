import csv
from mGST.reporting.reporting import random_seq_design
import numpy as np
from exp_design.exp_description import d

# Constants for gate sequence lengths
L_MIN = 1  # Shortest gate sequence
L_CUT = 8  # Cutoff: Half of the sequences are allocated to lengths < L_CUT
L_MAX = 14  # Maximum sequence length (l_max <= 24 due to implementation constraints)

# Prompt user for the number of gate sequences
N = int(input(
    'How many gate sequences would you like to generate? '
    'Recommended are at least 100 for a single qubit, '
    'at least 400 for two qubits, and at least 2000 for three qubits.\n >> '
))

# Calculate number of short and long sequences
N_short = int(np.ceil(N / 2))
N_long = int(np.floor(N / 2))

# Generate random sequences
J_all = random_seq_design(d, L_MIN, L_CUT, L_MAX, N_short, N_long)

# Reduce sequences to exclude negative values
J_reduced = [list(seq[seq >= 0]) for seq in J_all]

# Write sequences to CSV file
with open('exp_design/gen_sequences.csv', 'w', newline='') as myfile:
    writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    writer.writerows(J_reduced)

print('Sequences successfully saved in exp_design/gen_sequences.csv.')