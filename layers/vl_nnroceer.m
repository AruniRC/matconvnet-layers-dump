function [y, varargout] = vl_nnroceer(x1, x2, c, dzdy)
%VL_NNROCEER Calculate the verification EER between x1 and x2 in a batch
%
%   C       Labels: -1 for different, 1 for same. 1xBATCH_SZ
%   X1,X2   Pairwise features in correspondence: (X1_i,X2_i) form a pair.
%           1x1xKxBATCH_SZ.
%
%   Y       Output: EER over the batch. Scalar.
%   
%   Requires the VLFEAT toolbox for the `vl_roc()` function.
%
%   Note: this function does not back-propagate errors and cannot be used
%   as a training loss for the network. It is only for plotting the
%   accuracy of verification while training using the PairwiseLoss
%   objective. This is a very noisy approximation of the true EER since
%   this is calculated only over one batch at a time, then averaged over
%   batches.

backMode = ~isempty(dzdy);

useGpu = isequal(class(x1), 'gpuArray');

assert(numel(x1) == numel(x2));

if ~isequal(size(x1), size(x2))
    error('Reshape the matrices x1 and x2 to be the same shape.');
end

c = reshape(c, [1 1 1 numel(c)]);

if backMode
    y = zerosLike(x1);
    varargout{1} = zerosLike(x2);
else
    % forward pass
    d = sum((x1-x2).^2,3);  
    d = squeeze(d);
    y = get_val_eer(d, c);
%     if useGpu
%         y = gpuArray(y);
%     else
%         y = single(y);
%     end
end


% --------------------------------------------------------------------
function eer = get_val_eer(valDist, c)
% --------------------------------------------------------------------
% validation EER

[~,~,info] = vl_roc(c, -valDist);
eer = info.eer;


% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.zeros(size(x),classUnderlying(x)) ;
else
  y = zeros(size(x),'like',x) ;
end

% --------------------------------------------------------------------
function y = onesLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.ones(size(x),classUnderlying(x)) ;
else
  y = ones(size(x),'like',x) ;
end