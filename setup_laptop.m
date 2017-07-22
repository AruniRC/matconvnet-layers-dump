% add paths to libraries
VL_FEAT_PATH = '/Users/arunirc/Toolbox/vlfeat-0.9.20/toolbox/vl_setup';
MATCONVNET_PATH = '/Users/arunirc/Toolbox/matconvnet-1.0-beta23/matlab/vl_setupnn';


run(VL_FEAT_PATH);
run(MATCONVNET_PATH);

addpath(genpath('/Users/arunirc/Toolbox/matconvnet-1.0-beta23/examples'));
