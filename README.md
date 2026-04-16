the structure of the src folder: important files

we have 3 approaches, each we try with and without scheduler for learning rate => 6 approaches 
1, DANN:
baselineDANNOLD.py
baselineDANN.py
baselineDANNCosine.py

2, DANN in ensemble
ensembleDANNOld.py
ensembleDANN.py
ensembleDANNCosine.py

3, DANN in diversified ensemble
ensembleDANN2ViewOLD.py
ensembleDANN2View.py
ensembleDANN2ViewCosine.py

4, Author's scripts:
conv.py
dataset.py with its wrapper called synthetic_dataset.py
rmdn.py
train.py

5, results:
plot_ablation_study.py
full_ablation_study.png

