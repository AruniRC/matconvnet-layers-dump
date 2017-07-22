function [y, varargout] = vl_nnelemdiv(x1, x2, varargin)
%VL_NNELEMDIV Pointwise division of X1 by X2
%   MatConvNet layer equivalent of the MATLAB operation: x1 ./ x2 ;
%   If either x1(i) or x2(i) is small, then the output and the gradients at
%   the position (i) are set to zero. The threshold for "small" is set to
%   be THRESH = 1E-8.
%   X2 is allowed to be an array of same dimensions as X1 or a scalar.


hasBias = false;
backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
thresh = 1e-10;  % handle division by very small number
isScal = false;

if backMode
  dzdy = varargin{1} ;
end

x2_sz = size(x2);

% check dimensions
if size(x2,1)==1 && size(x2,2)==1 && size(x2,3)==1
    % if X2 is a scalar repmat to match X1 dimensions
    assert(isequal(size(x2,4), size(x1,4)));
    x2 = repmat(x2, [size(x1,1) size(x1,2) size(x1,3)]);
    isScal = true;
else    
    assert(isequal(numel(x1), numel(x2)));
    if ~isequal(size(x1), size(x2))
        x2 = reshape(x2, size(x1));
    end
    assert(isequal(size(x1), size(x2)));
end


% clip gradients if x1 or x2 is very small
clipIndex = (x1 < thresh) | (x2 < thresh) ;


if backMode
    y1 = 1 ./(x2 + thresh);
    y2 = - x1 ./(x2.^2 + thresh);
    y = y1 .* dzdy;
    dx2 = y2 .* dzdy;
    
    % clip small values
    y(clipIndex) = 0;
    dx2(clipIndex) = 0;
    
    % collapse gradient dimensions to the original scalar X2
    if isScal, dx2 = sum(sum(sum(dx2,1),2),3); end
    
    varargout{1} = dx2; 
else
    y = x1 ./ (x2 + thresh) ;
    y(clipIndex) = thresh;  % clip small values
end
