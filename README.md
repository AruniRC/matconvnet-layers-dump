# matconvnet-layers-dump

A (not final) resting place for neural network layers in MatConvNet. Some rigourously tested, others surviving on prayer and fasting.

### List of layers:

* Elementwise:
    - ElemDiv
    - ElemProd
* Losses:
    - LossRegul: regulariser on the features/activations. L1 or L2. 
    - L2Loss: Regression loss between targets and outputs.
    - PairwiseLoss: for verification
        + minimise distance between same pairs
        + maximise distance between different pairs
        + still testing ....
* Normalizers:
    - L2Norm: L2-normalize features to 1 in 2-norm. Taken from B-CNN project.
* Funky/Misc:
    - MixBasis: Form a linear combination of two branches of a network:
        + branch 1 gives a vector
        + branch 2 gives a matrix
        + output of MixBasis is the linear combination of the cols of the matrix using the elements of the vector as weights
    - BatchSplit: split the batch into two -- even and odd numbered samples. This can be pretty handy when training a siamese network for face verification.

### Example usage:

