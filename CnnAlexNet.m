outputFolder=fullfile('D:\GP\Image Classification by cnn\Fish Diseases');
rootFolder=fullfile(outputFolder,'Fish');
categories={'EUS1','ICH1','columnaris1','NormalFish'};
imds=imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tb1=countEachLabel(imds);
minSetCount=min(tb1{:,2});
label='';
net = alexnet;
[imdsTrain] = splitEachLabel(imds,0.7,'randomized');
net = alexnet
net.Layers;
analyzeNetwork(net);
imageSize=net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);

augimdsTrain = augmentedImageDatastore(imageSize(1:2),imdsTrain, ...
    'ColorPreprocessing','gray2rgb');

layer = 'fc7';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');

YTrain = imdsTrain.Labels;

%classifier = fitcecoc(featuresTrain,YTrain);






options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-3, ...
    'Verbose',false, ...
    'Plots','training-progress');



layer = 'fc7';
classifier=load ('classifier.mat')

layers = [ ...
    imageInputLayer([227 227 3])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer]
%trainedNet = trainNetwork(augimdsTrain,layers,options);

imageSize=[227,227,3];
D = 'D:\GP\Image Classification by cnn\Fish Diseases\Test';
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
newChr = extractBetween(name,1,1)
if(strcmp(string(label),'EUS1'))
      
accuracy=accuracy+1;
conn = database('phpmyadmin3','root','');
            c = date;

 colnames = {'date','farmhardwareid','disease'};
    insert(conn,'disease',colnames,{c,"aaaa","aaaaa"})

elseif (strcmp(string(label),'ICH1'))
   conn = database('phpmyadmin3','root','');
            c = date;

 colnames = {'date','farmhardwareid','disease'};
    insert(conn,'disease',colnames,{c,"aaaa","aaaaa"})

accuracy=accuracy+1;
elseif (strcmp(string(label),'columnaris1'))
accuracy=accuracy+1 ;
conn = database('phpmyadmin3','root','');
            c = date;

 colnames = {'date','farmhardwareid','disease'};
    insert(conn,'disease',colnames,{c,"aaaa","aaaaa"})

elseif(strcmp(string(label),'NormalFish')&&isequal(string(newChr),'4'))
accuracy=accuracy+1 ;

end
end
 disp('Accuracy')
 disp((accuracy/314)*100)
    