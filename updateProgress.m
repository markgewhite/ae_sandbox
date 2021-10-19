% *************************************
% Update the progress plots
% *************************************

function updateProgress( ax1, lineErr, scoreErr, ...
                         epoch, j, t0 )

addpoints( lineErr, j, ...
            double(extractdata(scoreErr)) );
        
        
% Update the title with training progress information.
D = duration( 0, 0, toc(t0), 'Format', 'hh:mm:ss' );
title( ax1, ...
    "Epoch: " + epoch + ", " + ...
    "Iteration: " + j + ", " + ...
    "Elapsed: " + string(D))
        

drawnow;

end