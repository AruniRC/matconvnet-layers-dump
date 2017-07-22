function [y, varargout] = vl_nnmixbasis(a, B, dzdy)
%   Weighted sum of bases B (matrix) with weights A (vector).

backMode = ~isempty(dzdy);

gpuMode = isa(a, 'gpuArray');

a = squeeze(a);
B = squeeze(B);

k = size(a,1);
d = size(B,1)/k;
bs = size(B,2);

if backMode
    % derivatives
    B = reshape(B, [k,d,bs]); % TODO - change to [d,k,bs]
    dzdy = squeeze(dzdy);
    da = zeros(k,bs, 'single');
    dB = zeros(k,d,bs, 'single');
    if gpuMode
        da = gpuArray(da);
        dB = gpuArray(dB);
    end
    
    % loop over batch :(
    for ii = 1:bs
        da(:,ii) = B(:,:,ii) * dzdy(:,ii);
        dB(:,:,ii) = a(:,ii) * dzdy(:,ii)';
    end
    
    y = reshape(da, [1 1 k bs]) ;
    varargout{1} = reshape(dB, [1 1 k*d bs]);
    
else
    % forward pass
    B = reshape(B, [k,d,bs]);
    y = zeros(d,bs, 'single');
    if gpuMode
        y = gpuArray(y);
    end
    
    % loop over batch :(
    for ii = 1:bs
        y(:,ii) = (a(:,ii)') * B(:,:,ii);
    end
    
    y = reshape(y, [1,1,d,bs]);
end