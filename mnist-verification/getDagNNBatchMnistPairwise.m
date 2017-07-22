function inputs = getDagNNBatchMnistPairwise(opts, imdb, batch)
%GETDAGNNBATCHPAIRWISE Forms training pairs from within the batch data

rng(0);
opts.shuffle = true; % TODO - expose option upstream

images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;  labels = single(labels);


% forming positive and negative pairs
numPairs = numel(labels);
assert(mod(numPairs,2)==0); % number of pairs,(= batch size), is even
posPair = getPosPairs(labels, unique(labels), numPairs/2); %2x(numPairs/2)
negPair = getNegPairs(labels, unique(labels), numPairs/2); %2x(numPairs/2) 


% arrange pairs as successive images
posIdx = posPair(:)';
negIdx = negPair(:)';
pairImages = images(:,:,:,[posIdx negIdx]) ;
pairLabels = [ones(1,numPairs/2) -ones(1,numPairs/2)];


% mean subtraction
% pairImages = bsxfun(@minus, pairImages, imdb.images.data_mean);

if isfield(opts, 'shuffle')    
    if opts.shuffle
        % shuffle pairs
        pairIdx = 1:2:size(pairImages,4); 
        p = randperm(length(pairIdx));
        pairIdx = pairIdx(p);

        % shuffle labels
        pairLabels = pairLabels(p);

        % sanity-check:
        %   p <= 50 ==> pairLabels = -1
        %   p >  50 ==> pairLabels = 1

        % shuffle images
        shuffleIdx = zeros(1, 2 * length(pairLabels));
        shuffleIdx(1:2:end) = pairIdx;
        shuffleIdx(2:2:end) = pairIdx+1;
        pairImages = pairImages(:,:,:,shuffleIdx);
    end
end

if opts.numGpus > 0
  pairImages = gpuArray(pairImages) ;
  pairLabels = gpuArray(pairLabels);
else
  pairImages = single(pairImages) ;
  pairLabels = single(pairLabels);  
end
% pairLabels = single(pairLabels); 
inputs = {'input', pairImages, 'label', pairLabels} ;


