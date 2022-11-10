% *********************************************
% MATLAB Adversarial Autoencoder - MNIST images 
% *********************************************

clear;

% Parameters

doUseGPU = false;

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

setup.ae.fullCalc = false;

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

% setup the minibatch queues that don't require parallel processing
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

% --- setup parallel processing ---

% setup the environment
if canUseGPU && doUseGPU
    executionEnvironment = "gpu";
    numberOfGPUs = gpuDeviceCount('available');
    delete(gcp('nocreate'));
    pool = parpool(numberOfGPUs);
else
    executionEnvironment = 'cpu';
    delete(gcp('nocreate'));
    pool = parpool;
end
numWorkers = pool.NumWorkers;

% set the minibatch sizes
miniBatchSize = setup.ae.batchSize;
if executionEnvironment == "gpu"
    % scale-up batch proportional to the number of workers
    miniBatchSize = miniBatchSize .* numWorkers;
end

% determine batch size per worker
wkMiniBatchSize = floor(miniBatchSize ./ repmat(numWorkers,1,numWorkers));
remainder = miniBatchSize - sum(wkMiniBatchSize);
wkMiniBatchSize = wkMiniBatchSize + [ones(1,remainder) zeros(1,numWorkers-remainder)];

% find indices of the mean and variance state parameters 
% of the batch normalization layers in the network state property.
% so that mean and variance can be aggregated across all workers
[isEncBNStateMean, isEncBNStateVar] = ...
                    batchNormalizationStateLayers( dlnetEnc );
[isDecBNStateMean, isDecBNStateVar] = ...
                    batchNormalizationStateLayers( dlnetDec );

% create a data queue to allow training to be terminated early, 
% if user presses the stop button
spmd
    stopTrainingEventQueue = parallel.pool.DataQueue;
end
stopTrainingQueue = stopTrainingEventQueue{1};

% start the timer
start = tic;

% completet the monitoring setup
dataQueue = parallel.pool.DataQueue;
displayFcn =  @(data) updateProgressAE( data, ...
                                lossAx, lineReconLoss, lineAuxLoss, start );
afterEach( dataQueue, displayFcn );

nIter = floor( size(trainX,3)/setup.ae.batchSize );
i = 0;
stopRequest = false;

% begin the parallel training - each worker runs this code
spmd

    % partition datastore to divide it up among the workers
    wkDSTrain = partition( imgDSTrain, numWorkers, spmdIndex );

    % create minibatchqueue using partitioned datastore on each worker
    wkMbqTrain = minibatchqueue(  wkDSTrain,...
                            'MiniBatchSize', wkMiniBatchSize(spmdIndex), ...
                            'PartialMiniBatch', 'discard', ...
                            'MiniBatchFcn', @preprocessMiniBatch, ...
                            'MiniBatchFormat', {'CB', 'CB'} );

    epoch = 0;
    while epoch < setup.ae.nEpochs && ~stopRequest

        epoch = epoch + 1;

        % Shuffle the data
        shuffle( wkMbqTrain );
    
        % loop over mini-batches
        while spmdReduce(@and, hasdata(wkMbqTrain)) && ~stopRequest            
            i = i + 1;
            
            % Read mini-batch of data
            [wkXTrain, wkYTrain] = next( wkMbqTrain );
    
            % generate density estimation
            if setup.ae.fullCalc
                dlPTrain = dlarray(calcXDistribution( extractdata(wkXTrain) ), 'CB');
            else
                dlPTrain = [];
            end
    
            % Evaluate the model gradients and the generator state using
            % dlfeval and the modelGradients function listed at the end of the
            % example.
            [ wkGradEnc, wkGradDec, wkStateEnc, wkStateDec, wkLoss ] = ...
                                      dlfeval(  @modelGradientsAE, ...
                                                dlnetEnc, ...
                                                dlnetDec, ...
                                                wkXTrain, ...
                                                dlPTrain, ...
                                                setup.ae.fullCalc );

            % Aggregate the losses on all workers
            wkNormFactor = wkMiniBatchSize(spmdIndex)./miniBatchSize;
            loss = spmdPlus( wkNormFactor*extractdata(wkLoss) );

            % Aggregate the network states across all workers
            dlnetEnc.State = aggregateState( wkStateEnc, ...
                                             wkNormFactor, ...
                                             isEncBNStateMean, ...
                                             isEncBNStateVar );
            dlnetDec.State = aggregateState( wkStateDec, ...
                                             wkNormFactor, ...
                                             isDecBNStateMean, ...
                                             isDecBNStateVar );

            % Aggregate the gradients across all workers
            wkGradEnc.Value = dlupdate( @aggregateGradients, ...
                                        wkGradEnc.Value, ...
                                        {wkNormFactor} );
            wkGradDec.Value = dlupdate( @aggregateGradients, ...
                                        wkGradDec.Value, ...
                                        {wkNormFactor} );
    
            % Update the decoder network parameters
            [ dlnetDec, avgG.dec, avgGS.dec ] = ...
                                adamupdate( dlnetDec, ...
                                            wkGradDec, ...
                                            avgG.dec, ...
                                            avgGS.dec, ...
                                            i, ...
                                            setup.dec.learnRate, ...
                                            setup.ae.beta1, ...
                                            setup.ae.beta2 );
            
            % Update the generator network parameters
            [ dlnetEnc, avgG.enc, avgGS.enc ] = ...
                                adamupdate( dlnetEnc, ...
                                            wkGradEnc, ...
                                            avgG.enc, ...
                                            avgGS.enc, ...
                                            i, ...
                                            setup.enc.learnRate, ...
                                            setup.ae.beta1, ...
                                            setup.ae.beta2 );
    
        end

        % Stop training if the Stop button has been clicked
        stopRequest = spmdPlus(stopTrainingEventQueue.QueueLength);

        % Send training progress information to the client
        if spmdIndex == 1
            data = [epoch, i, loss];
            send( dataQueue, gather(data) );
        end

    end

end

% Every validationFrequency iterations, 
% display batch of generated images 
% using the held-out generator input.
%if mod( i, setup.ae.valFreq ) == 0 || i == 1
    % validationUpdate( mbqTest, dlnetEnc, dlnetDec, wkLoss )
%end    