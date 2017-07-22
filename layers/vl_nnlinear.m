function [y, varargout] = vl_nnlinear(x1, x2, varargin)
%VL_NNELEMPROD Implement WX + b, where [W,b] is x1 and X is x2
%   Detailed explanation goes here



backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

assert(isequal(numel(x1), numel(x2)));
if ~isequal(size(x1), size(x2))
    x2 = reshape(x2, size(x1));
end

% assert(isequal(size(x1), size(x2)));

if backMode
    y1 = x2;
    y2 = x1;
    y = y1 .* dzdy;
    varargout{1} = y2 .* dzdy;
else
    y = x1 .* x2;
end
