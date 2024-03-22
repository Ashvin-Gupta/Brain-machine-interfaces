function [modelParameters] = positionEstimatorTraining2(training_data, dt)

    %Calculating spiking Rates and hand position velocity
    % Arrays to hold the spiking rates,firing rates and x and y velocity 
    spikingR = [];
    firingR = [];
    xVelocity = [];
    yVelocity = [];
    trainingData = struct([]);
    velXY = struct([]);
    %dt = 10; % bin size
    

    % Knn Classifier (used to predict reaching angle from the first 320ms)
    % Arrays to hold the spike train and reaching angle when doing KNN
    spikes = [];
    reachingAngle = [];
    spikeCount = zeros(length(training_data),98);

    % Loop over all directions
    for direction = 1:8
        % Loop over all neurons 
        for i = 1:98
            % Loop over spike train
            for trial = 1:length(training_data)
                % Store the number of spikes 
                number_of_spikes = length(find(training_data(trial,direction).spikes(i,1:320)==1));
                spikeCount(trial,i) = number_of_spikes;
                
                % Account for the fact that movement begins after 300ms 
                for t = 300:dt:550-dt

                    % find the firing rates of one neural unit per trial
                    spike_train = training_data(trial, direction).spikes(i, t:t+dt);
                    numSpikes = sum(spike_train == 1);
                    spikingR = cat(2, spikingR, numSpikes/(dt*0.001));

                    % Find the velocity of the hand movement
                    % Only has to be done once per trial
                    if i==1
                        x_vel = (training_data(trial,direction).handPos(1,t+dt) - training_data(trial,direction).handPos(1,t)) / (dt*0.001);
                        y_vel = (training_data(trial,direction).handPos(2,t+dt) - training_data(trial,direction).handPos(2,t)) / (dt*0.001);
                        xVelocity = cat(2, xVelocity, x_vel);
                        yVelocity = cat(2, yVelocity, y_vel);
                    end

                end
                % store firing rate of one neural unit for every trial in one array
                firingR = cat(2, firingR, spikingR);
                spikingR = [];

            end
            trainingData(i,direction).firingR = firingR;
            velXY(direction).x = xVelocity;
            velXY(direction).y = yVelocity;
            firingR = [];

        end
        % Calculate spikes and reaching anlge for KNN
        spikes = cat(1, spikes, spikeCount);
        reaching_angle(1:length(training_data)) = direction;
        reachingAngle = cat(2, reachingAngle, reaching_angle);

        xVelocity = [];
        yVelocity = [];
    end

    % Train KNN model
    K = 18; % number of neighbors to consider
    knnModel.spikes = spikes;
    knnModel.reachingAngle = reachingAngle;
    knnModel.K = K;
    
   

    % Linear Regression to predict velocity
    velModel= struct([]);
    for direction = 1:8
        vel = [velXY(direction).x; velXY(direction).y];
        firingR = [];
        for i = 1:98
            firingR = cat(1, firingR, trainingData(i,direction).firingR);
        end


        velModel(direction).reachingAngle = lsqminnorm(firingR',vel');
    end

    % Store trained model parameters
    modelParameters = struct('velModel', velModel, 'knnModel', knnModel);

end
