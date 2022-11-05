% *********************************************
% MATLAB Adversarial Autoencoder - MNIST images 
% *********************************************

clear;

if ~ismac
    cd 'C:\Users\m.g.e.white\My Drive\Academia\MATLAB'
end

% Parameters

rng('default');

% AAE training parameters
setup.ae.nEpochs = 50; 
setup.ae.batchSize = 1000;
setup.ae.beta1 = 0.5;
setup.ae.beta2 = 0.999;
setup.ae.valFreq = 100;
setup.ae.testSize = 1000;
setup.ae.dispSize = [2 5];

setup.ae.zDim = 10;
setup.ae.xDim = [ 28 28 1 ];

setup.ae.fullCalc = true;

% encoder network parameters
setup.enc.learnRate = 0.0002;
setup.enc.scale = 0.2;
setup.enc.input = prod( setup.ae.xDim );
setup.enc.output = setup.ae.zDim;

% decoder network parameters
setup.dec.learnRate = 0.0002;
setup.dec.scale = 0.2;
setup.dec.input = setup.ae.zDim;
setup.dec.output = prod( setup.ae.xDim );

% Load Data
load('mnistAll.mat');
trainX = mnist.train_images;
trainY = mnist.train_labels;

imgDSTrainX = arrayDatastore( trainX, 'IterationDimension', 3 );
imgDSTrainY = arrayDatastore( trainY );
imgDSTrain = combine( imgDSTrainX, imgDSTrainY );

testX = mnist.test_images; 
testY = mnist.test_labels;

imgDSTestX = arrayDatastore( testX, 'IterationDimension', 3 );
imgDSTestY = arrayDatastore( testY );
imgDSTest = combine( imgDSTestX, imgDSTestY );

% define the networks
[ dlnetEnc, dlnetDec ] = aeDesign1( setup );

% train the model
% ---------------

mbqTrain = minibatchqueue(  imgDSTrain,...
                            'MiniBatchSize', setup.ae.batchSize, ...
                            'PartialMiniBatch', 'discard', ...
                            'MiniBatchFcn', @preprocessMiniBatch, ...
                            'MiniBatchFormat', {'CB', 'CB'} );
mbqTest = minibatchqueue(  imgDSTest,...
                            'MiniBatchSize', setup.ae.testSize, ...
                            'PartialMiniBatch', 'discard', ...
                            'MiniBatchFcn', @preprocessMiniBatch, ...
                            'MiniBatchFormat', {'CB', 'CB'} );
mbqDisp = minibatchqueue(  imgDSTest,...
                            'MiniBatchSize', prod(setup.ae.dispSize), ...
                            'PartialMiniBatch', 'discard', ...
                            'MiniBatchFcn', @preprocessMiniBatch, ...
                            'MiniBatchFormat', {'CB', 'CB'} );

% initialise training parameters
avgG.enc = []; 
avgG.dec = []; 
avgGS.enc = [];
avgGS.dec = [];

% initialise training plots
f = figure;
f.Position(3) = 2*f.Position(3);

imgOrigAx = subplot( 2, 2, 1 );
imgReconAx = subplot( 2, 2, 3 );
distAx = subplot( 2, 2, 2 );
lossAx = subplot( 2, 2, 4 );

lineReconLoss = animatedline( lossAx, 'Color', [0.4940, 0.1840, 0.5560] );
lineAuxLoss = animatedline( lossAx, 'Color', [0.8500 0.3250 0.0980] );

legend( lossAx, {'Recon', 'Aux'} );
%ylim( errorAx, [0 0.1] );
xlabel( lossAx, "Iteration");
ylabel( lossAx, "Loss");
grid on;

legend( distAx, 'Latent Distribution' );
xlabel( distAx, 'Z');
ylabel( distAx, 'Q(Z)');

% train the GAN 
% -------------

nIter = floor( size(trainX,3)/setup.ae.batchSize );
start = tic;
j = 0;
% Loop over epochs.
for epoch = 1:setup.ae.nEpochs
    
    % Shuffle the data
    shuffle( mbqTrain );

    % Loop over mini-batches.
    for i = 1:nIter
        
        j = j + 1;
        
        % Read mini-batch of data
        [dlXTrain, dlYTrain] = next( mbqTrain );

        % generate density estimation
        dlPTrain = dlarray(calcXDistribution( extractdata(dlXTrain) ), 'CB');
        dlXTrain = dlarray( extractdata(dlXTrain), 'CB');

        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [ gradEnc, gradDec, reconLoss, dlZTrain ] = ...
                                  dlfeval(  @modelGradientsAE, ...
                                            dlnetEnc, ...
                                            dlnetDec, ...
                                            dlXTrain, ...
                                            dlPTrain, ...
                                            setup.ae.fullCalc );

        % Update the decoder network parameters
        [ dlnetDec, avgG.dec, avgGS.dec ] = ...
                            adamupdate( dlnetDec, ...
                                        gradDec, ...
                                        avgG.dec, ...
                                        avgGS.dec, ...
                                        j, ...
                                        setup.dec.learnRate, ...
                                        setup.ae.beta1, ...
                                        setup.ae.beta2 );
        
        % Update the generator network parameters
        [ dlnetEnc, avgG.enc, avgGS.enc ] = ...
                            adamupdate( dlnetEnc, ...
                                        gradEnc, ...
                                        avgG.enc, ...
                                        avgGS.enc, ...
                                        j, ...
                                        setup.enc.learnRate, ...
                                        setup.ae.beta1, ...
                                        setup.ae.beta2 );
                
        % Every validationFrequency iterations, 
        % display batch of generated images 
        % using the held-out generator input.
        if mod( i, setup.ae.valFreq ) == 0 || i == 1

            % fit auxiliary model
            ZTrain = double(extractdata(dlZTrain))';
            YTrain = double(extractdata(dlYTrain));
            auxModel = fitcecoc( ZTrain, YTrain );

            % test auxiliary model
            if ~hasdata( mbqTest )
                shuffle( mbqTest )
            end
            [dlXTest, dlYTest] = next( mbqTest );

            dlZTest = predict( dlnetEnc, dlXTest );
            ZTest = double(extractdata(dlZTest))';
            YTest = double(extractdata(dlYTest));
            YTestPred = predict( auxModel, ZTest );
            auxLoss = crossentropy( YTestPred, YTest );

            updateImagesPlot( imgOrigAx, imgReconAx, ...
                              dlnetEnc, dlnetDec, ...
                              dlXTest, setup.ae );
            %( 'PostDoc/Examples/AE/Networks/AE Networks WIP.mat', ...
            %      'dlnetEnc', 'dlnetDec' );
            
            if ~hasdata( mbqDisp )
                shuffle( mbqDisp );
            end
            dlXDist = next( mbqDisp );
            dlZDist = predict( dlnetEnc, dlXDist );
            updateDistPlot( distAx, dlZDist );

            disp([ 'Loss (' num2str(epoch) ') = ' num2str(reconLoss) ]);

        end          

        updateProgressAE(   lossAx, ...
                            lineReconLoss, lineAuxLoss, ...
                            reconLoss, auxLoss, ...
                            epoch, j, start );
    end
end

