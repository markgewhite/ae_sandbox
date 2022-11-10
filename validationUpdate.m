function validationUpdate( nets, mbqTest, mbqDisp, ...
                           imgOrigAx, imgReconAx, distAx, setup )

    dlnetEnc = nets(1);
    dlnetDec = nets(2);

    % get test data
    if ~hasdata( mbqTest )
        shuffle( mbqTest )
    end
    [dlXTest, dlYTest] = next( mbqTest );

    dlZTest = predict( dlnetEnc, dlXTest );
    ZTest = double(extractdata(dlZTest))';

    if setup.ae.fullCalc

        % fit auxiliary model
        ZTrain = double(extractdata(dlZTrain))';
        YTrain = double(extractdata(wkYTrain));
        auxModel = fitcecoc( ZTrain, YTrain );

        YTest = double(extractdata(dlYTest));
        YTestPred = predict( auxModel, ZTest );
        auxLoss = crossentropy( YTestPred, YTest );

    else
        auxLoss = 0;

    end
        
    updateImagesPlot( imgOrigAx, imgReconAx, ...
                      dlnetEnc, dlnetDec, ...
                      dlXTest, setup.ae );
    
    if ~hasdata( mbqDisp )
        shuffle( mbqDisp );
    end
    dlXDist = next( mbqDisp );
    dlZDist = predict( dlnetEnc, dlXDist );
    updateDistPlot( distAx, dlZDist );

end