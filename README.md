# matconvnet-layers-dump

A (not so final) resting place for neural network layers in MatConvNet. Most rigourously tested, a few surviving on prayer, fasting and MNIST test alone.

### List of layers:

* Elementwise:
    - ElemDiv
    - ElemProd
* Losses:
    - LossRegul: regulariser on the features/activations. L1 or L2. 
    - L2Loss: Regression loss between targets and outputs.
    - PairwiseLoss: for verification or siamese networks
        + minimise distance between same pairs
        + maximise distance between different pairs
        + tested on MNIST
* Normalizers:
    - L2Norm: L2-normalize features to 1 in 2-norm. Taken from [B-CNN](https://bitbucket.org/tsungyu/bcnn.git) codebase.
* Funky/Misc:
    - MixBasis: Form a linear combination of two branches of a network:
        + branch 1 gives a vector
        + branch 2 gives a matrix
        + output of MixBasis is the linear combination of the cols of the matrix using the elements of the vector as weights
    - BatchSplit: split the batch into two -- even and odd numbered samples. This can be pretty handy when training a siamese network for face verification.


### Source code organization:

Each `vl_*.m` file implements the logic of the forward and backward passes for a layer. Each class file wraps a particular `vl_*.m` function so that they can be used to define a DAG object in MatConvNet. 


### Example usage: TODO


### Adding a conv layer in DAG:
    % add a fixed layer that sums over all the channels
    %   -- pred = sum_i(w_i*x_i) + b
    sum_conv = dagnn.Conv('size',[1 1 785 1],'pad',0,'stride',1,'hasBias',false);
    net.addLayer('sum_conv', sum_conv, {'wx'}, {'pred'}, {'sum_conv_f', 'sum_conv_b'});
    net.params(net.getParamIndex('sum_conv_f')).value = ones(1, 1, 785, 1, 'single');
    net.params(net.getParamIndex('sum_conv_f')).learningRate = 0;
    net.params(net.getParamIndex('sum_conv_b')).value = single(0);
    net.params(net.getParamIndex('sum_conv_b')).learningRate = 0; 
