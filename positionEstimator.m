function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)

    tMin = length(test_data.spikes)-20;
    tMax = length(test_data.spikes);

    % Get trained model parameters
    knnModel = modelParameters.knnModel;
    velModel = modelParameters.velModel;

    % Initialize variables
    % set direction once using first 320ms
    spikeCount = [];
    if tMax <= 320
        spikeCount = sum(test_data.spikes(:, 1:320) == 1, 2);
        %spikeCount = zscore(spikeCount);
        direction = knnPredict(knnModel, spikeCount);
    else
        direction = modelParameters.direction; % use already set direction
    end
    
    % Make KNN prediction

    %firing rate
    firingRate = sum(test_data.spikes(:, tMin:tMax), 2) ./ (20 * 0.001);
    
    % velocity
    velModel = modelParameters.velModel(direction).reachingAngle;
    Vx = dot(firingRate, velModel(:, 1));
    Vy = dot(firingRate, velModel(:, 2));
    
    % new position
    if tMax <= 320
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
    else
        lastPos = test_data.decodedHandPos(:, end);
        x = lastPos(1) + Vx * (20 * 0.001);
        y = lastPos(2) + Vy * (20 * 0.001);
    end

    % update model parameters
    newModelParameters.velModel = modelParameters.velModel;
    newModelParameters.knnModel = modelParameters.knnModel;
    newModelParameters.direction = direction;

end

function [predictedLabel] = knnPredict(knnModel, spikes)
    %train_spikes = zscore(knnModel.spikes);

    % Calculate pairwise Euclidean distances between test and training data
    distances = sqrt(sum((knnModel.spikes - spikes.').^2, 2));
    
    % Find the K nearest neighbors
    [~, indices] = sort(distances);
    neighbors = knnModel.reachingAngle(indices(1:knnModel.K));
    
    % Predict the class label as the mode of the nearest neighbors
    predictedLabel = mode(neighbors);

end






