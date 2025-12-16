# Competing Speeds of Memorisation and Generalisation Predict Grokking

Deep networks trained with gradient descent often exhibit grokking on small algorithmic datasets, where training accuracy saturates quickly while test accuracy remains near chance for many steps before achieving perfect generalisation.
Existing hypotheses usually appeal to _information-theoretic capacity_: the network overfits and memorises before discovering a compact algorithmic solution; or to differences in _pattern learning speeds_ across features: the network learns fast features that generalise poorly first.
In this work, we present a theory that unifies both perspectives to make quantitative predictions for when grokking should occur, demonstrating that the onset of grokking can be attributed to an optimisation tradeoff between the _speed_ with which a model memorises random data and the _speed_ with which it discovers an algorithmic solution.
We establish information-theoretic estimates of model capacity and dataset complexity, confirming that models generalise when the two are comparable. Surprisingly, even with a capacity that is theoretically greater than dataset complexity, we find that smaller models prefer to generalise first, with grokking occurring only for larger models. We argue that this is because smaller models have lower memorisation speeds, so gradient descent finds the generalising solution first. We show empirically that the onset of grokking can be predicted where generalising speed intersects with memorisation speed.
Our results support a picture in which (i) model capacity determines when both memorising and algorithmic solutions are representable, while (ii) their _relative learning speeds_ determine which solution dominates. This gives rise to three regimes of learning as model size increases: underfitting, immediate generalisation, and grokking.
We argue that memorisation capacity and learning speeds are sufficient, in principle, to analytically predict these regimes for modular arithmetic, and outline how this programme could be extended to larger models and natural tasks.


## License and Attribution

Unless otherwise stated, the files and code in this repository are licensed under the GNU GENERAL PUBLIC LICENSE (Version 3), Copyright (C) 2025 Yiding Song and Hanming Ye.

**Note:** the files [`data.py`](data.py) and [`models.py`](models.py) are adapted from the code by Amund Tveit (available at [adveit/torch_grokking](https://github.com/atveit/torch_grokking) under the MIT License), which itself is a PyTorch port of the original MLX code by Jason Stock (available at [stockeh/mlx-grokking](https://github.com/stockeh/mlx-grokking)). I have modified `data.py` to add different split types and random data generation, and left `models.py` untouched. The trainers used to finetune the neural network also takes inspiration from the code of Tveit and Stock.
