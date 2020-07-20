
# Folders content

1. ./rabi_bangrampVsCrab_comparison and ./old_rabi_crab_results:
	Not sure, this is old stuff.

2. ./rabi_and_lmg_optimizations_201902XX:
	For Rabi and LMG perform optimization for several different total times, trying with different ansatze (doublebang, bangramp, crab with 2, 3 and 10 frequencies), different optimization methods (nelder-mead, powell, cg).
	There are some problems in the 17 and 27 results, so use results of 28 as reference.
	The optimizations in the 28 are performed constraining the parameters to double the critical parameter.

3. ./rabi_and_lmg_optimizations_different_constraints_20190228:
	As above, but changing how much the parameters are constrained.

4. ./parameters_scans_20190221:
	Fixing a constant function protocol, for both rabi and lmg, we scan many possible values of total time and height of the protocol and look at how the fidelity varies.
	Also contains Mathematica notebook to analyse the results.

5. ./lmg_different_spinnumbers_20190228:
	Same as the scans in rabi_and_lmg_optimiations, but only for the lmg, fixing the parameter constraints, and trying with many different total numbers of spins.
