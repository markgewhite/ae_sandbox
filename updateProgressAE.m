% *************************************
% Update the progress plots
% *************************************

function updateProgressAE(  data, ax1, lineRecon, lineAux, t0 )

% extra data from the queue
epoch = data(1);
i = double(data(2));
reconLoss = double(data(3));
auxLoss = 0;

addpoints( lineRecon, i, reconLoss );

%addpoints( lineAux, j, auxLoss );
        
% Update the title with training progress information.
D = duration( 0, 0, toc(t0), 'Format', 'hh:mm:ss' );
title( ax1, ...
    "Epoch: " + epoch + ", " + ...
    "Iteration: " + i + ", " + ...
    "Elapsed: " + string(D))
        

drawnow;

end