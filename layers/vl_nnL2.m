function y = vl_nnL2(x, c, dzdy)
%VL_NNL2 Calculate the L2 loss between x and the targets c

backMode = ~isempty(dzdy);

assert(numel(x) == numel(c));

if ~isequal(size(x), size(c))
    c = reshape(c, size(x));
end

if backMode
    % derivatives
    y = +((x-c)) * dzdy;
    y = reshape(y, size(x));
else
    % forward pass
    y = 0.5*sum((x-c).^2);
    y = sum(y(:)); % sum over the batch
end