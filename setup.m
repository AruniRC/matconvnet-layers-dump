% add paths to libraries
VL_FEAT_PATH = '/home/aroychow/Toolbox/vlfeat-0.9.20/toolbox/vl_setup';
MATCONVNET_PATH = '/home/aroychow/Toolbox/matconvnet-1.0-beta24/matlab/vl_setupnn';

run(VL_FEAT_PATH);
run(MATCONVNET_PATH);

addpath(genpath('./edge_predict'));
addpath(genpath('./cnn_filter_predict'));
addpath('./rectangle_predict')
% addpath('./vis-face-track/');
% addpath('./layers/');
% addpath('./occluded-mnist/');
% addpath(genpath('./util/'));
% addpath('./cnn_train/');
% addpath('./Ncut_9/');  % normalised cuts library

addpath(genpath('/home/aroychow/Toolbox/matconvnet-1.0-beta24/examples'));
