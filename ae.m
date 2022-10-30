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
setup.ae.batchSize = 100;
setup.ae.beta1 = 0.5;
setup.ae.beta2 = 0.999;
setup.ae.valFreq = 100;
setup.ae.valSize = [2 5];
setup.ae.distSize = 1000;

setup.ae.zDim = 10;
setup.ae.xDim = [ 28 28 1 ];

setup.ae.singleStage = false;

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
testX = mnist.test_images; 
testY = mnist.test_labels;

imgDSTrain = arrayDatastore( trainX, 'IterationDimension', 3 );
imgDSTest = arrayDatastore( testX, 'IterationDimension', 3 );

% define the networks
[ dlnetEnc, dlnetDec ] = aeDesign1( setup );

% train the model
% ---------------

mbqTrain = minibatchqueue(  imgDSTrain,...
                            'MiniBatchSize', setup.ae.batchSize, ...
                            'PartialMiniBatch', 'discard', ...
                            'MiniBatchFcn', @preprocessMiniBatch, ...
                            'MiniBatchFormat', 'CB' );
mbqTest = minibatchqueue(  imgDSTest,...
                            'MiniBatchSize', prod( setup.ae.valSize ), ...
                            'PartialMiniBatch', 'discard', ...
                            'MiniBatchFcn', @preprocessMiniBatch, ...
                            'MiniBatchFormat', 'CB' );
mbqDist = minibatchqueue(  imgDSTest,...
                            'MiniBatchSize', setup.ae.distSize, ...
                            'PartialMiniBatch', 'discard', ...
                            'MiniBatchFcn', @preprocessMiniBatch, ...
                            'MiniBatchFormat', 'CB' );

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
errorAx = subplot( 2, 2, 4 );

lineScoreErr = animatedline( errorAx, ...
                                    'Color', [0.4940, 0.1840, 0.5560] );

legend( errorAx, 'Reconstruction Error' );
ylim( errorAx, [0 0.1] );
xlabel( errorAx, "Iteration");
ylabel( errorAx, "MSE");
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
        dlXTrain = next( mbqTrain );
        
        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [ gradEnc, gradDec, scoreErr, dlZTrain, dlXHatTrain ] = ...
                                  dlfeval(  @modelGradientsAE, ...
                                            dlnetEnc, ...
                                            dlnetDec, ...
                                            dlXTrain, ...
                                            setup.ae.singleStage );

        % perform the second stage separately, if required
        if ~setup.ae.singleStage
            [ gradEnc, gradDec, scoreErr ] = ...
                                      dlfeval(  @modelGradientsAE2, ...
                                                dlnetEnc, ...
                                                dlnetDec, ...
                                                dlXTrain, ...
                                                dlZTrain, ...
                                                dlXHatTrain );
        end

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
            if ~hasdata( mbqTest )
                shuffle( mbqTest )
            end
            dlXTest = next( mbqTest );
            updateImagesPlot( imgOrigAx, imgReconAx, ...
                              dlnetEnc, dlnetDec, ...
                              dlXTest, setup.ae );
            save( 'PostDoc/Examples/AE/Networks/AE Networks WIP.mat', ...
                  'dlnetEnc', 'dlnetDec' );
            
            if ~hasdata( mbqDist )
                shuffle( mbqDist );
            end
            dlXDist = next( mbqDist );
            dlZDist = predict( dlnetEnc, dlXDist );
            updateDistPlot( distAx, dlZDist );

            disp([ 'Loss (' num2str(i) ') = ' num2str(scoreErr) ]);

        end
        
        updateProgressAE( errorAx, lineScoreErr, scoreErr, ...
                        epoch, j, start );
    end
end

