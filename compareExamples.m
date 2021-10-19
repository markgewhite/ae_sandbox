% ************************************************************
% Compare examples from standard and adversarial autoencoders
% ************************************************************

clear;

% parameters
imageSize = [ 28 28 1 ];
nCodes = 10; % number of codes
nPixels = prod( imageSize );

% load trained networks
if ~ismac
    cd 'C:\Users\m.g.e.white\My Drive\Academia\MATLAB'
end

% load the adversarial autoencoder
load( 'PostDoc/Examples/AAE/Networks/AAE Networks v1.3.mat' );
dlnetEncAAE = dlnetEnc;
dlnetDecAAE = dlnetDec;

% load the standard autoencoder
load( 'PostDoc/Examples/AE/Networks/AE Networks v1.0.mat' );
dlnetEncAE = dlnetEnc;
dlnetDecAE = dlnetDec;

clear 'dlnetEnc' 'dlnetDec';

% load data
load('mnistAll.mat');
testX = mnist.test_images; 
testY = mnist.test_labels;

imgDSTest = arrayDatastore( testX, 'IterationDimension', 3 );

% extract batches of 20
mbq = minibatchqueue(  imgDSTest,...
                           'MiniBatchSize', 20, ...
                           'PartialMiniBatch', 'discard', ...
                           'MiniBatchFcn', @preprocessMiniBatch, ...
                           'MiniBatchFormat', 'CB' );

% setup plots
fObj = figure;
fObj.Position(3) = 2*fObj.Position(3);

axOrig = subplot(3,1,1);
axAE = subplot(3,1,2);
axAAE = subplot(3,1,3);

while hasdata( mbq )

    % select random example
    shuffle( mbq );
    dlX = next( mbq );
    
    % encode batch with AE and AAE
    dlZae = predict( dlnetEncAE, dlX );
    dlZaae = predict( dlnetEncAAE, dlX );
    
    % reconstruct images
    dlXae = predict( dlnetDecAE, dlZae );
    dlXaae = predict( dlnetDecAAE, dlZaae );
    
    dlX = reshape( dlX, imageSize(1), imageSize(2), [] );
    dlXae = reshape( dlXae, imageSize(1), imageSize(2), [] );
    dlXaae = reshape( dlXaae, imageSize(1), imageSize(2), [] );

    % generate tiled output
    imgOrig = imtile( extractdata( dlX ), 'GridSize', [ 2 10 ] );
    imgOrig = rescale( imgOrig, 0, 255 );

    imgAE = imtile( extractdata( dlXae ), 'GridSize', [ 2 10 ] );
    imgAE = rescale( imgAE, 0, 255 );

    imgAAE = imtile( extractdata( dlXaae ), 'GridSize', [ 2 10 ] );
    imgAAE = rescale( imgAAE, 0, 255 );
    
    % Display the images.
    image( axOrig, imgOrig );
    colormap gray;
    title( axOrig, 'Original Images' );
    xticklabels( axOrig, [] );
    yticklabels( axOrig, [] );

    image( axAE, imgAE );
    colormap gray;
    title( axAE, 'AE-reconstructed Images' );
    xticklabels( axAE, [] );
    yticklabels( axAE, [] );

    image( axAAE, imgAAE );
    colormap gray;
    title( axAAE, 'AAE-reconstructed Images' );
    xticklabels( axAAE, [] );
    yticklabels( axAAE, [] );
    
    pause;
    
end


