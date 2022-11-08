% *************************************
% Update the images plot
% *************************************

function updateImagesPlot( ax1, ax2, ...
                           dlnetEnc, dlnetDec, ...
                           dlXVal, setup )

dlZVal = predict( dlnetEnc, dlXVal );
dlXValHat = predict( dlnetDec, dlZVal );

% reshape the images
dlXVal = reshape( dlXVal, setup.xDim(1), setup.xDim(2), [] );
dlXValHat = reshape( dlXValHat, setup.xDim(1), setup.xDim(2), [] );

% Tile and rescale the images in the range [0 1].
valImg1 = imtile( extractdata( dlXVal ), 'GridSize', setup.dispSize );
valImg1 = rescale( valImg1, 0, 255 );

valImg2 = imtile( extractdata( dlXValHat ), 'GridSize', setup.dispSize );
valImg2 = rescale( valImg2, 0, 255 );

% Display the images.
image( ax1, valImg1 );
colormap gray;
title( ax1, "Original Images" );
xticklabels( ax1, [] );
yticklabels( ax1, [] );

image( ax2, valImg2 );
colormap gray;
title( ax2, "Reconstructed Images" );
xticklabels( ax2, [] );
yticklabels( ax2, [] );

drawnow;

end