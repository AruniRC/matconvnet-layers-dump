function [net, info] = cnn_mnist_verif(varargin)
%CNN_MNIST_VERIF    Train CNN on MNIST using verification loss     


opts.batchNormalization = false ;
opts.networkType = 'simplenn' ;
opts.baseNetwork = [] ;
opts.modelSet = {};
opts.fixedOccluders = false; % occlusions are pre-applied, fixed
opts.modelType = 'pairwise';
opts.occlusionType = 'none';
opts.fixLayers = true;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['mnist-baseline-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;



% --------------------------------------------------------------------
%                                                         Prepare net
% --------------------------------------------------------------------

% Load the network
%   If a base network is provided, then don't initialize from scratch
switch lower(opts.networkType)
    case 'simplenn'
        if isempty(opts.baseNetwork)
            net = cnn_mnist_init('batchNormalization', opts.batchNormalization, ...
                     'networkType', opts.networkType) ;
        else
            net = load(opts.baseNetwork); 
            net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
            net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
              {'prediction', 'label'}, 'error') ;
            net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
              'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
        end
  
    case 'dagnn'
        if isempty(opts.baseNetwork)
            net = cnn_mnist_init('batchNormalization', opts.batchNormalization, ...
                     'networkType', opts.networkType) ; 
                 
        else
            if ischar(opts.baseNetwork)
                net = dagnn.DagNN.loadobj(opts.baseNetwork);
            else
                net = opts.baseNetwork;
            end
        end
    
    otherwise
        assert(false) ;
end   
% -------------------------------------------------------------------------
% net.move('cpu');




% ------------------------------------------------------------------------- 
% Modify network for verification loss
% -------------------------------------------------------------------------
if isa(net.layers(end).block, 'VerifEER') || isa(net.layers(end).block, 'PairwiseLoss')
    % if fine-tuned net, trained earlier with pairwise loss
    net.removeLayer('roc_eer');
    net.removeLayer('p_loss');
    net.removeLayer('bsplit');
    net.removeLayer('l2n');
    
else
    % if baseline net, pre-trained with softmax loss
    net.removeLayer('top1err');
    net.removeLayer('top5err');
    net.removeLayer('layer8');
    net.removeLayer('layer7');
end


if opts.fixLayers
   for ii = 1:length(net.params)
       net.params(ii).learningRate = 0;
       net.params(ii).weightDecay = 0;
   end  
end


switch opts.modelType
   
    case 'pairwise'
        % vanilla pairwise siamese-network
        net.addLayer('l2n', L2Norm(), {'x6'}, {'l2_1'});
        net.addLayer('bsplit', BatchSplit(), {'l2_1'}, {'f1', 'f2'});
        net.addLayer('p_loss', PairwiseLoss(), {'f1', 'f2', 'label'}, {'objective'});
        net.addLayer('roc_eer', VerifEER(), {'f1', 'f2', 'label'}, {'EER'});
        
    case {'binary-mask', 'occlusion-mask'}
        % 
        f = 1/100;
         
        % L2-normalize and split the fc-layer activations
        net.addLayer('l2n', L2Norm(), {'x6'}, {'l2_1'});
        net.addLayer('bsplit_f', BatchSplit(), {'l2_1'}, {'f1', 'f2'});
        
        
        % -----------------------------------------------------------------
        % Binarizing branch 
        %   conv from last fully-conv layer
        b_conv = dagnn.Conv('size',[4,4,50,500],'pad',0,'stride',1,'hasBias',true);
        net.addLayer('b_conv', b_conv, {'x4'}, {'x5_b'}, {'b_conv_f', 'b_conv_b'});
        net.params(net.getParamIndex('b_conv_f')).value = f*randn(4,4,50,500, 'single');
        net.params(net.getParamIndex('b_conv_f')).learningRate = 1;
        net.params(net.getParamIndex('b_conv_b')).value = zeros(1,500,'single');
        net.params(net.getParamIndex('b_conv_b')).learningRate = 1; 
        
        % binarize
        net.addLayer('b_bin', Binarize(), {'x5_b'}, {'b_1'});      % -1/1
        net.addLayer('b_relu', dagnn.ReLU() , {'b_1'}, {'b_2'});  %  0/1 
        
        % batch split
        net.addLayer('b_split', BatchSplit(), {'b_2'}, {'bf1', 'bf2'});
        
        % binary mask AND - positions that are ON in both binary vectors
        net.addLayer('b_and', ElemProd(), {'bf1', 'bf2'}, {'b_mask'});
        % -----------------------------------------------------------------          
        
        
        % Elementwise multiply binary mask with fc features
        %   for both splits of the batch
        net.addLayer('elemprod_1', ElemProd(), {'b_mask', 'f1'}, {'g1'});
        net.addLayer('elemprod_2', ElemProd(), {'b_mask', 'f2'}, {'g2'});
        
        
        net.addLayer('p_loss', PairwiseLoss(), {'g1', 'g2', 'label'}, {'objective'});
        net.addLayer('roc_eer', VerifEER(), {'g1', 'g2', 'label'}, {'EER'});   
        
    otherwise
        error('Unknown model type.');
end


%{

%%
net.mode = 'test' ;
img = rand(28,28,1,10);
for ii = 1:length(net.vars)
    net.vars(ii).precious = true;
end


%%
opts.numGpus = 0;
opts.shuffle = false;
batch = 1:50;
inputs = getDagNNBatchMnistPairwise(opts, imdb, batch);
net.eval(inputs);

net.eval({'input', single(img)});
%}



% if ~isempty(opts.train.gpus)
%    move(net, 'gpu'); 
% end

net.meta.trainOpts.learningRate = opts.train.learningRate;
net.meta.trainOpts.numEpochs =  opts.train.numEpochs;




% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  
  switch opts.occlusionType
    case 'none'
        
    case 'occlusion-mask'
        
        % Occlude left or right half of alternate images
        leftHalf = zeros(28, 28, 'single'); leftHalf(1:28, 1:14) = 1;
        % rightHalf = zeros(28, 28, 'single'); rightHalf(1:28, 15:28) = 1;
        occlusion_mask = {leftHalf};
        
        % apply ONLY left-half occlusion to ALL images
        %   (see if the binary mask learns to ignore these portions)
        data1 = apply_halfplane_mask(imdb.images.data(:,:,:,1:2:end), ... 
                                     0, occlusion_mask{1}, 128);
                                 
%         data2 = apply_halfplane_mask(imdb.images.data(:,:,:,2:2:end), ... 
%                                      0, occlusion_mask{2}, 128);
        imdb.images.data(:,:,:,1:2:end) = data1 ;
        % imdb.images.data(:,:,:,2:2:end) = data2 ;
  end
  
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end




% net.meta.classes.name = {'3', '5'} ;



% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end
 
    
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% -------------------------------------------------------------------------



% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatchMnistPairwise(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;



% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
