outputFolder=fullfile('C:\Users\Owner\Desktop\Image Classification by cnn\Fish Diseases');
rootFolder=fullfile(outputFolder,'Fish');
categories={'EUS1','ICH1','columnaris1','NormalFish'};
imds=imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tb1=countEachLabel(imds);
minSetCount=min(tb1{:,2});

imds=splitEachLabel(imds,minSetCount,'randomize');
countEachLabel(imds);
EUS=find(imds.Labels=='EUS1',1);
ICH=find(imds.Labels=='ICH1',1);
NormalFish=find(imds.Labels=='NormalFish',1);
columnaris=find(imds.Labels=='columnaris1',1);
figure
subplot(2,2,1);
imshow(readimage(imds,EUS));

subplot(2,2,2);
imshow(readimage(imds,ICH));

subplot(2,2,3);
imshow(readimage(imds,columnaris));

subplot(2,2,4);
imshow(readimage(imds,NormalFish));


[imdsTrain] = splitEachLabel(imds,0.7,'randomized');
net = inceptionresnetv2();
net.Layers;
analyzeNetwork(net);
imageSize=net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);

augimdsTrain = augmentedImageDatastore(imageSize(1:2),imdsTrain, ...
    'ColorPreprocessing','gray2rgb');

layer = 'ClassificationLayer_predictions';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;

classifier = fitcecoc(featuresTrain,YTrain);


layers = [ ...
    imageInputLayer([ 299   299     3])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer]



options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-1, ...
    'Verbose',false, ...
    'Plots','training-progress');
trainedNet = trainNetwork(augimdsTrain,layers,options);




D = 'C:\Users\Owner\Desktop\Image Classification by cnn\Fish Diseases\Test';
S = dir(fullfile(D,'*.png'));
 accuracy=1;
for k = 1:numel(S)
   
     F = fullfile(D,S(k).name);
   
    newImage = imread(F);
    imshow(newImage);
    
  
    
    ds=augmentedImageDatastore(imageSize,...
newImage ,'ColorPreprocessing','gray2rgb');

imageFeatures=activations(net,...
    ds,layer,'MiniBatchSize',32,'OutputAs','columns');

label=predict(classifier,imageFeatures,'ObservationsIn','columns');

sprintf('The loaded image belongs to %s class',label)

name=S(k).name;
newChr = extractBetween(name,1,1);









if(strcmp(string(label),'EUS1')&&isequal(string(newChr),'2'))
      
accuracy=accuracy+1;

elseif (strcmp(string(label),'ICH1')&&isequal(string(newChr),'3'))
   
accuracy=accuracy+1;

elseif (strcmp(string(label),'columnaris1')&&isequal (string(newChr),'1'))
accuracy=accuracy+1 ;


elseif(strcmp(string(label),'NormalFish')&&isequal(string(newChr),'4'))
accuracy=accuracy+1 ;


end

end
 disp('Accuracy')
 disp((accuracy/314)*100)
    