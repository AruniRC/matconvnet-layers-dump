classdef Linear  < dagnn.ElementWise
    %ELEMMULTREGUL Elementwise multiplication
    %   Elem-wise multiplication of two inputs:  X1 .* X2
    
    properties
        
    end
    
    methods
        
        function outputs = forward(obj, inputs, params)
          outputs{1} = vl_nnlinear(inputs{1}, inputs{2}) ;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
          [derInputs{1}, derInputs{2}] =  ...
              vl_nnlinear(inputs{1}, inputs{2}, derOutputs{1}) ;
          derParams = {} ;
        end        
        
        function rfs = getReceptiveFields(obj)
          rfs.size = [1 1] ;
          rfs.stride = [1 1] ;
          rfs.offset = [1 1] ;
        end

        function obj = Linear(varargin)
          obj.load(varargin) ;
        end
        
        
    end
    
end

