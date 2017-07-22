function [x1, varargout] = vl_nnbatchsplit(x, dzdy)
%VL_NNBATCHSPLIT Splits the batch into two  halves.
%   Useful for verification, where two images or samples would be paired
%   data. VL_NNSPLITBATCH() can separate these into two streams of output,
%   X1 and X2. These can be fed into the input to a pairwise loss function.

backMode = ~isempty(dzdy);
% isGpu = isequal(class(x), 'gpuArray');

assert(isequal(mod(size(x,4),2),0)); % batchsize must be an even number

if backMode
    % backward pass
    dy1 = dzdy{1};
    dy2 = dzdy{2};
    sz = size(dy1);
    
    x1 = zeros([sz(1:3) 2*sz(4)], class(dy1));  
    x1(:,:,:,1:2:end) = dy1;
    x1(:,:,:,2:2:end) = dy2;
else
    % forward pass
    x1 = x(:,:,:,1:2:end);
    varargout{1} = x(:,:,:,2:2:end);
end

