function y = vl_nnlossregul(x, dzdy, type)
%VL_NNLOSSREGUL Calculate the loss from regularising a feature X
%   Only supports L1 (LASSO) regulariser and L2 (ridge) for now.
%   The alpha or rate of decay is to be set in the opts.derOutputs{}
%   options for the loss during the call to cnn_train().
%   N.B. - the alpha works only for DagNN currently.

% TODO - different types: L1 and L2 regularizations

backMode = ~isempty(dzdy);
type = upper(type);

if backMode
    % derivatives
    switch type
        case 'L1'
            y = bsxfun(@times, sign(x), dzdy) ; 
        case 'L2'
            y = 2 * bsxfun(@times, x, dzdy) ; 
    end
else
    % sum the loss over spatial and feature (channel) dimensions
    %   -- also sums over batches
    switch type
        case 'L1'
            y = sum(sum(sum(sum(abs(x), 1), 2), 3),4);
        case 'L2'
            y = sum(sum(sum(sum(x.^2, 1), 2), 3),4);
    end
end

