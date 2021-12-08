clc
close all

path = 'C:\Users\munna\OneDrive\Desktop\Math Lab Tumor Code\munna'

data = fullfile(path, 'flower');
train = imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames'); 

count = train.countEachLabel;

net = alexnet;

inputSize = net.Layers(1).InputSize(1:2); 
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);

layers = [imageInputLayer([256 256 3])
 net(2:end-3)
 fullyConnectedLayer(2)
 softmaxLayer
 classificationLayer()
]

opt = trainingOptions('sgdm','Maxepoch',10,'InitialLearnRate',0.0001)
training = trainNetwork(train,layers,opt)


im = imread('C:\Users\munna\OneDrive\Desktop\Math Lab Tumor Code\munna\m21.jpg');


out = classify(training, im);

figure,imshow(im)
title(string(out))



