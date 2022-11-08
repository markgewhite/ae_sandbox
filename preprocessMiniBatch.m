% *************************************
% Preprocess images
% *************************************

function [X, Y] = preprocessMiniBatch( XInput, YInput )

% concatenate mini-batch
X = cat( 3, XInput{:} );
Y = cat( 1, YInput{:} );

% rescale the images in the range [-1 1].
X = rescale( X, -1, 1, 'InputMin', 0, 'InputMax', 255 );

% reshape into a long vector
X = reshape( X, numel(X(:,:,1)), [] );

end