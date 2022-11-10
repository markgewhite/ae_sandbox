function [isStateMean, isStateVar] = batchNormalizationStateLayers( net )
    % Identify which network layers are batch normalization states

    bnLayers = arrayfun( @(l) isa(l,"nnet.cnn.layer.BatchNormalizationLayer"), ...
                                    net.Layers );
    
    names = string({net.Layers(bnLayers).Name});

    isStateMean = ismember( net.State.Layer, names ) ...
                            & net.State.Parameter == "TrainedMean";
    isStateVar  = ismember( net.State.Layer, names ) ...
                            & net.State.Parameter == "TrainedVariance";

end