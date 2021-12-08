
clc
close all

path = 'C:\Users\munna\OneDrive\Desktop\Math Lab Tumor Code\munna'

data = fullfile(path, 'flower'); 
imds = imageDatastore(data, 'LabelSource', 'foldernames', 'IncludeSubfolders',true); 

tbl = countEachLabel(imds);

minSetCount = min(tbl{:,2}); 
maxNumImages = 100;   % change
minSetCount = min(maxNumImages,minSetCount);
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)

net = resnet50();
net.Layers(1)

[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb')
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% testing image start; loading image

testimage = imageDatastore({'m21.JPG'});
orginalImage = readimage(testimage,1);

% K-mean clustering start

I = im2double(orginalImage);                    % Load Image
F = reshape(I,size(I,1)*size(I,2),3);                 % Color Features
%% K-means
K     = 8;                                            % Cluster Numbers
CENTS = F( ceil(rand(K,1)*size(F,1)) ,:);             % Cluster Centers
DAL   = zeros(size(F,1),K+2);                         % Distances and Labels
KMI   = 10;                                           % K-means Iteration
for n = 1:KMI
   for i = 1:size(F,1)
      for j = 1:K  
        DAL(i,j) = norm(F(i,:) - CENTS(j,:));      
      end
      [Distance, CN] = min(DAL(i,1:K));               % 1:K are Distance from Cluster Centers 1:K 
      DAL(i,K+1) = CN;                                % K+1 is Cluster Label
      DAL(i,K+2) = Distance;                          % K+2 is Minimum Distance
   end
   for i = 1:K
      A = (DAL(:,K+1) == i);                          % Cluster K Points
      CENTS(i,:) = mean(F(A,:));                      % New Cluster Centers
      if sum(isnan(CENTS(:))) ~= 0                    % If CENTS(i,:) Is Nan Then Replace It With Random Point
         NC = find(isnan(CENTS(:,1)) == 1);           % Find Nan Centers
         for Ind = 1:size(NC,1)
         CENTS(NC(Ind),:) = F(randi(size(F,1)),:);
         end
      end
   end
end
X = zeros(size(F));
for i = 1:K
idx = find(DAL(:,K+1) == i);
X(idx,:) = repmat(CENTS(i,:),size(idx,1),1); 
end
T = reshape(X,size(I,1),size(I,2),3);

% k-means clustering end


% feature Extract using wavelet start

C=[];
D = 'C:\Users\munna\OneDrive\Desktop\Math Lab Tumor Code\munna';
filePattern = fullfile(D, '*.jpg');
jpegFiles = dir(filePattern);
numfiles = length(jpegFiles);

for k = 1:length(jpegFiles)
  baseFileName = jpegFiles(k).name;
  fullFileName = fullfile(D, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  
imageArray = imread(fullFileName);

signal1 = imageArray(:,:);
[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3 cH3 cV3 cD3];
C = [C(:);DWT_feat(:)];
drawnow; 
end

%imagesc(C);

% feature Extract using wavelet end

ds = augmentedImageDatastore(imageSize,orginalImage, 'ColorPreprocessing', 'gray2rgb');
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

predictedLabel = predict(classifier,imageFeatures, 'ObservationsIn', 'columns')

% testing image End

figure
imshow(im2bw(T,.6))
title(string(predictedLabel))

%%k flod validation start

path = 'C:\Users\munna\OneDrive\Desktop\Math Lab Tumor Code\munna';

data = fullfile(path, 'flower');
train = imageDatastore(data, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

count = train.countEachLabel;

res=0;

%for k = 1:10
[imdsTrain, imdsValidation] = splitEachLabel(train, 0.30, 'randomized');
net = alexnet;

layers = [imageInputLayer([256 256 3])
 net(2:end-3)
 fullyConnectedLayer(2)
 softmaxLayer
 classificationLayer()
];

opt = trainingOptions('sgdm', 'MiniBatchSize', 4 , 'Maxepoch', 15 , 'InitialLearnRate', 0.001, 'Plots','training-progress');
training = trainNetwork(imdsTrain, layers, opt);

num = numel(imdsTrain.Labels);
idx = randperm(num, 50);

TP=0;
TN=0;
FP=0;
FN=0;

%figure
for i = 1:50
    
    %subplot(4,4,i)
    [I, info] = readimage(imdsTrain, i);
    imshow(I)
    str1 = string(info.Label);

    out = classify(training, I);
    str2 = string(out);
    
    ss = strcat(str1, "  ");
    
    str = strcat(ss, str2);
    
    %title(str)
    
    if str1 == "Benign"
        if str2 == "Benign"
            TP=TP+1;
        else
            FN=FN+1;
        end
        
    else 
        if str2 == "Malignant"
            TN=TN+1;
        else
            FP=FP+1;
        end
    end
    
end

res = res + ((TP+TN)/(TP+TN+FP+FN))*100;

%res=res/10;

fprintf('Acuracy = %0.2f%%\n', res);

