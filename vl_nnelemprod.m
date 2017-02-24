function [y, varargout] = vl_nnelemprod(x1, x2, varargin)
%VL_NNELEMPROD Summary of this function goes here
%   Detailed explanation goes here


hasBias = false;
backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

% if used to have [x; 1]
if abs(size(x1,1)*size(x1,2)*size(x1,3) ... 
        - size(x2,1)*size(x2,2)*size(x2,3)) == 1
    szImg = size(x2);
    x2 = reshape(x2, [1 1 size(x2,1)*size(x2,2) size(x2,4)]);
    x2 = cat(3, x2, ones(1, 1, 1, size(x2,4), 'single')) ;
    hasBias = true;
end

assert(isequal(numel(x1), numel(x2)));
if ~isequal(size(x1), size(x2))
    x2 = reshape(x2, size(x1));
end

assert(isequal(size(x1), size(x2)));

if backMode
    y1 = x2;
    y2 = x1;
    y = y1 .* dzdy;
    dx2 = y2 .* dzdy;
    if hasBias
        dx2 = dx2(:,:,1:end-1,:);
        dx2 = reshape(dx2, szImg);
    end
    varargout{1} = dx2; 
else
    y = x1 .* x2;
end
