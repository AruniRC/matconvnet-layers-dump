function y = vl_nnmatmult(a, b, dzdy)
% VL_NNMATMULT

backMode = ~isempty(dzdy);


if backMode
    % derivatives
    
else
    % forward pass
    y = a * b;
end