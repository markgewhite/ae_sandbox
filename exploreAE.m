% *********************************************
% MATLAB Adversarial Autoencoder - Explorer
% *********************************************

clear;

% parameters
imageSize = [ 28 28 1 ];
nCodes = 10; % number of codes
codeRange = -2.0:0.2:2.0;
nSteps = length( codeRange );
nPixels = prod( imageSize );

% load trained networks
load( 'PostDoc/Examples/AE/Networks/AE Networks v1.0.mat' );

% load data
load('mnistAll.mat');
testX = mnist.test_images; 
testY = mnist.test_labels;

imgDSTest = arrayDatastore( testX, 'IterationDimension', 3 );

% extract 1000 random samples
mbqRange = minibatchqueue(  imgDSTest,...
                           'MiniBatchSize', 1000, ...
                           'PartialMiniBatch', 'discard', ...
                           'MiniBatchFcn', @preprocessMiniBatch, ...
                           'MiniBatchFormat', 'CB' );
shuffle( mbqRange );
dlXRange = next( mbqRange );
dlZRange = predict( dlnetEnc, dlXRange );

% obtain mean and standard deviation
dlZMean = mean( dlZRange, 2 );
dlZStd = std( dlZRange, 0, 2 );
                       
mbqTest = minibatchqueue(  imgDSTest,...
                           'MiniBatchSize', 1, ...
                           'PartialMiniBatch', 'discard', ...
                           'MiniBatchFcn', @preprocessMiniBatch, ...
                           'MiniBatchFormat', 'CB' );

figure;
ax = subplot( 1, 1, 1 );
                       
dlX = dlarray(zeros( nCodes*nSteps*nPixels, 1 ));
while hasdata( mbqTest )

    % select random example
    shuffle( mbqTest );
    dlXTest = next( mbqTest );
    
    % encode it
    dlZ0 = predict( dlnetEnc, dlXTest );
    
    c = 0;
    for i = 1:nCodes
        
        % reset
        dlZ = dlZ0;
        
        for j = codeRange
        
            % vary one code
            dlZ(i) = dlarray( dlZ0(i) + j*dlZStd(i) );
            
            % update counters
            c = c + 1;
            
            % decode the image
            dlX( (c-1)*nPixels+1:c*nPixels ) = predict( dlnetDec, dlZ );
            
        end
        
    end
    
    dlX = reshape( dlX, imageSize(1), imageSize(2), [] );
    
    img = imtile( extractdata( dlX ), 'GridSize', [ nCodes nSteps ] );
    img = rescale( img, 0, 255 );
    
    % Display the images.
    image( ax, img );
    colormap gray;
    xticklabels( ax, [] );
    yticklabels( ax, [] );
    
    pause;
    
end

