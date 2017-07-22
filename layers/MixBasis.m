
classdef MixBasis < dagnn.Filter
%   Weighted mixture of basis matrices:
%       Given K basis matrices as inputs{2} and mixtures weights as a
%       vector input{2}, this DagNN wrapper class computes their weighted 
%       sum.

  properties
    normalizeGradients = false;
  end

  methods
    function outputs = forward(obj, inputs, params)
        outputs{1} = vl_nnmixbasis(inputs{1}, inputs{2}, []);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [derInputs{1}, derInputs{2}] = vl_nnmixbasis(inputs{1}, inputs{2}, derOutputs{1});

      if obj.normalizeGradients
          for i=1:numel(derInputs)
              gradNorm = sum(abs(derInputs{i}(:))) + 1e-8;
              derInputs{i} = derInputs{i}/gradNorm;
          end
      end
      derParams = {} ;
    end
    
    function rfs = getReceptiveFields(obj)
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = MixBasis(varargin)
      obj.load(varargin) ;
    end
  end
end


