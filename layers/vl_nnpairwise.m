function [y, varargout] = vl_nnpairwise(x1, x2, c, dzdy)
%VL_NNPAIRWISELOSS Calculate the pairwise loss between x1 and x2
%
%   C       Labels: -1 for different, 1 for same. 1xBATCH_SZ
%   X1,X2   Pairwise features in correspondence: (X1_i,X2_i) form a pair.
%           1x1xKxBATCH_SZ.
%
%   Y       Output: sum of squared errors. 1xBATCH_SZ.
%   
%   Minimizes L2 distance between X1 and X2 for C=1 and maximizes distance  
%   for C=-1.
%   L2-normalizing X1, X2 may be a good idea.

backMode = ~isempty(dzdy);

assert(numel(x1) == numel(x2));

if ~isequal(size(x1), size(x2))
    error('Reshape the matrices x1 and x2 to be the same shape.');
end

c = reshape(c, [1 1 1 numel(c)]);

if backMode
    % assert(isequal(size(x1), size(dzdy)));
    
    % derivatives
    y = +((x1-x2)) .* dzdy;   % dy/dx1
    y = bsxfun(@times, y, c);    
    varargout{1} = - y;       % dy/dx2
else
    % forward pass
    y = 0.5*sum((x1-x2).^2,3);     
    y = bsxfun(@times, y, c); 
    
    % sum over batches to give single scalar output for objective
    %   hack: easier plotting error.
    y = sum(y(:));
end
