% *********************************************
%  Fully-connected type AE design
% *********************************************

function [ dlnetEnc, dlnetDec ] = aeDesign1( setup )

% define the encoder network
layersEnc = [
    featureInputLayer( setup.enc.input, 'Name', 'in' ) 
    fullyConnectedLayer( 512, 'Name', 'fc1' )
    leakyReluLayer( setup.enc.scale, 'Name', 'lrelu1' )
    fullyConnectedLayer( 512, 'Name', 'fc2' )
    leakyReluLayer( setup.enc.scale, 'Name', 'lrelu2' )
    fullyConnectedLayer( setup.enc.output, 'Name', 'fc3' )
    leakyReluLayer( setup.enc.scale, 'Name', 'lrelu3' )
    ];

lgraphEnc = layerGraph( layersEnc );
dlnetEnc = dlnetwork( lgraphEnc );


% define the decoder network
layersDec = [
    featureInputLayer( setup.dec.input, 'Name', 'in' )    
    fullyConnectedLayer( 512, 'Name', 'fc1' )
    leakyReluLayer( setup.dec.scale, 'Name', 'lrelu1' )
    fullyConnectedLayer( 512, 'Name', 'fc2' )
    leakyReluLayer( setup.dec.scale, 'Name', 'lrelu2' )
    fullyConnectedLayer( prod(setup.dec.output), 'Name', 'fc3' )
    leakyReluLayer( setup.dec.scale, 'Name', 'lrelu3' )
    tanhLayer( 'Name', 'out' );
    ];

lgraphDec = layerGraph( layersDec );
dlnetDec = dlnetwork( lgraphDec );


end