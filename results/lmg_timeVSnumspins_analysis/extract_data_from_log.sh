grep -o '\(Iteration\|fidelity\|spins\|Starting\).*' ./goodTime_vs_numSpins_precise.log | perl -pe 's/.*tf=(.*)\n/\1 /' |  perl -pe 's/fidelity: (.*)/\1/' | perl -pe 's/Iteration (.*)\/.*\n/\1 /' | perl -pe 's/Starting.*?([0-9]+).*/\1/' > readableLog.txt