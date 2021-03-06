
outputFolder=fullfile('C:\Users\Owner\Desktop\Image Classification by cnn\Fish Diseases\YCBCR');
rootFolder=fullfile(outputFolder,'Fish');
categories={'1-YCBCR-EUS','2-YCBCR-ICH','3-YCBCR-columnaris','4-YCBCR-NormalFish'};
imds=imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tb1=countEachLabel(imds);
minSetCount=min(tb1{:,2});

imds=splitEachLabel(imds,minSetCount,'randomize');
countEachLabel(imds);
EUS=find(imds.Labels=='1-YCBCR-EUS',1);
ICH=find(imds.Labels=='2-YCBCR-ICH',1);
NormalFish=find(imds.Labels=='4-YCBCR-NormalFish',1);
columnaris=find(imds.Labels=='3-YCBCR-columnaris',1);

figure
subplot(2,2,1);
imshow(readimage(imds,EUS));

subplot(2,2,2);
imshow(readimage(imds,ICH));

subplot(2,2,3);
imshow(readimage(imds,columnaris));

subplot(2,2,4);
imshow(readimage(imds,NormalFish));










net = resnet18();
figure
plot(net);
title('ResNet-18');
set(gca,'YLim',[150 170]);
net.Layers(1)
net.Layers(end);

numel(net.Layers(end).ClassNames);
analyzeNetwork(net)





[trainingSet]=splitEachLabel(imds,0.7,'randomize');
imageSize=net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);

augmentedTrainingSet=augmentedImageDatastore(imageSize(1:2),...
trainingSet,'ColorPreprocessing','gray2rgb');


% augmentedTestSet=augmentedImageDatastore(imageSize(1:2),...
% Testset,'ColorPreprocessing','gray2rgb');






featureLayer='fc1000';
trainingFeatures=activations(net,...
    augmentedTrainingSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');


trainingLabeles=trainingSet.Labels;
classifier=fitcecoc(trainingFeatures,trainingLabeles,...
    'Learner','Linear','Coding','onevsall','ObservationsIn','columns');


% testFeatures=activations(net,...
%     augmentedTestSet,featureLayer,'MiniBatchSize',32,'OutputAs','columns');
% 
% predictLabels=predict(classifier,testFeatures,'ObservationsIn','columns');
% testLables=Testset.Labels;


% testImage = readimage(Testset,1);
% testLabel = Testset.Labels(1);


% condMat=confusionmat(testLables,predictLabels);
% confMat=bsxfun(@rdivide,condMat,sum(condMat,2));
% mean(diag(confMat));


% label=[];
% for index=1:length(label)
%     label(index)=1;
%     index=index+1;
% end
% 
% 
% YPred = classify(trainedNet,newImage);
% accuracy=0;
% for i=1:length(YPred)
%     if(YPred(i)==label(i))
%         accuracy=accuracy+1;
%     end
% end
        
% 
layers = [ ...
    imageInputLayer(imageSize)
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer]

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-3, ...
    'Verbose',false, ...
    'Plots','training-progress');


trainedNet = trainNetwork(trainingSet,layers,options);






D = 'C:\Users\Owner\Desktop\Image Classification by cnn\Fish Diseases\YCBCR\YCBCR-Test';
S = dir(fullfile(D,'*.png'));
 accuracy=1;
for k = 1:numel(S)
   
     F = fullfile(D,S(k).name);
   
    newImage = imread(F);
    imshow(newImage);
    
  
    
    ds=augmentedImageDatastore(imageSize,...
newImage ,'ColorPreprocessing','gray2rgb');

imageFeatures=activations(net,...
    ds,featureLayer,'MiniBatchSize',32,'OutputAs','columns');

label=predict(classifier,imageFeatures,'ObservationsIn','columns');

sprintf('The loaded image belongs to %s class',label)

name=S(k).name;
newChr = extractBetween(name,1,1);









if(strcmp(string(label),'1-YCBCR-EUS')&&isequal(string(newChr),'2'))
      
accuracy=accuracy+1

elseif (strcmp(string(label),'2-YCBCR-ICH')&&isequal(string(newChr),'3'))
   
accuracy=accuracy+1

elseif (strcmp(string(label),'3-YCBCR-columnaris')&&isequal (string(newChr),'1'))
accuracy=accuracy+1 


elseif(strcmp(string(label),'4-YCBCR-NormalFish')&&isequal(string(newChr),'4'))
accuracy=accuracy+1 


end

end

 disp('Accuracy')
 disp((accuracy/314)*100)
    










