classdef BatchSplit < dagnn.ElementWise
    %BATCHSPLIT Split an input batch into two halves
    
    properties
        
    end
    
    methods
        
        function outputs = forward(obj, inputs, params)
          [outputs{1}, outputs{2}] = vl_nnbatchsplit(inputs{1}, []) ;
        end

        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
          derInputs{1} = vl_nnbatchsplit(inputs{1}, derOutputs) ;
          derParams = {} ;
        end        
        
        function rfs = getReceptiveFields(obj)
          rfs.size = [1 1] ;
          rfs.stride = [1 1] ;
          rfs.offset = [1 1] ;
        end

        function obj = BatchSplit(varargin)
          obj.load(varargin) ;
        end
        
        
    end
    
end

