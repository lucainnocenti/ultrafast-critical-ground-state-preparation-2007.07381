This directory contains the results of optimisations performed in order to establish the functional relationship between the time required to get a good fidelity (more than some fixed threshold), and the number of spins of the LMG model under consideration.

To do this, for each value of the total number of spins, we optimise for a double-bang protocol for many possible tf, stopping when we reach a time corresponding to a good fidelity.

The datasets `goodTime_vs_numSpins.csv` and `goodTime_vs_numSpins_saveall.csv` both contain the same results, with the only difference that in `goodTime_vs_numSpins_saveall.csv` all of the parameters necessary to reconstruct the found optimal protocol have also been saved (I probably forgot to do it in the other dataset).

The dataset `goodTime_vs_numSpins_coarse.csv` contains the results obtained by scanning the total evolution times with a coarser grid (it’s an obsolete dataset).


- `goodTime_vs_numSpins_precise.csv`
This is a rerun of the `goodTime_vs_numSpins.csv` dataset, using optimisation options to try and get more reliable results. We also set up the target fidelity as 0.999 here. This seems to have had a notable effect in the results. It appears to be much easier (in terms of time required) to get >0.99 fidelities than it is to get >0.999 ones.
NOTE: The maximum scan time (tf=2) was set as too low here, as for spin numbers greater than 130 we always see the optimisation stop at tf=2.0 without the target fidelity being reached.
NOTE: There was also a bug in the way the results are saved: the second column contains the fidelities, not the final time tf.


- `data_extracted_from_precise_csv.txt`
This is the dataset of fidelities vs times for many different numbers of spins, extracted from the log file of the `goodTime_vs_numSpins_precise.csv` dataset with the following command:
	grep -o '\(Iteration\|fidelity\|spins\|Starting\).*' ./goodTime_vs_numSpins_precise.log | perl -pe 's/.*tf=(.*)\n/\1 /' |  perl -pe 's/fidelity: (.*)/\1/' | perl -pe 's/Iteration (.*)\/.*\n/\1 /' | perl -pe 's/Starting.*?([0-9]+).*/\1/' > readableLog.txt