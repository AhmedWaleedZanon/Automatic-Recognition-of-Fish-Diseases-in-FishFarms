Detector = vision.CascadeObjectDetector('stopSignDetector.xml');
outputFolder=fullfile('C:\Users\ASUS\Desktop\GP\Image Classification by cnn\Fish Diseases\Test');
rootFolder=fullfile(outputFolder,'Fish');
categories={'1-XYZ-EUS','2-XYZ-ICH','3-XYZ-columnaris','4-XYZ-NormalFish'};
%D = 'C:\Users\Owner\Desktop\Image Classification by cnn\Fish Diseases\XYZ\XYZTest';
videoFileReader = VideoReader('testvideo1.avi');
folder = 'C:\Users\ASUS\Desktop\GP\Image Classification by cnn\CroppedImage';
Resizefolder = 'C:\Users\ASUS\Desktop\GP\Image Classification by cnn\Resize';
TestFolder = 'C:\Users\ASUS\Desktop\GP\Image Classification by cnn\TestFolder';
RenameFolder = 'C:\Users\ASUS\Desktop\GP\Image Classification by cnn\RenameFolder';
videoFrame = readFrame(videoFileReader); 
Detector.MinSize = [63,279];


Detector.MergeThreshold =20;
bbox =Detector(videoFrame);
numFrames = (videoFileReader.FrameRate*videoFileReader.Duration);
detectFish(videoFileReader,folder,bbox);
Resize(folder,Resizefolder);
Rename(Resizefolder,RenameFolder);
ColorSpace(RenameFolder,TestFolder);
 TrainDisease(rootFolder,categories,TestFolder);
videoframes(numFrames,bbox);




function detectFish(videoFileReader,folder,bbox)
ii=0;
while hasFrame(videoFileReader)
videoFrame = readFrame(videoFileReader); 

videoFrames = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Fish');
for j=1:size(bbox,1)
 FishCrop=imcrop(videoFrame,bbox(j,1:4));
 baseFileName = sprintf('%d.png', ii); % e.g. "1.png"
fullFileName = fullfile(folder, baseFileName); % No need to worry about slashes now!
imwrite(FishCrop, fullFileName);
 segmention(fullFileName);
ii=ii+1;
end
n=size(bbox,1);
str_n=num2str(n);
 str=strcat(str_n,' fish Detect');
 disp(str);
	pause(1/videoFileReader.FrameRate);
end
end


function Resize(folder,Resizefolder)
S = dir(fullfile(folder,'*.png'));
for k = 1:numel(S)
    F = fullfile(folder,S(k).name);
    imycc = imread(F);
RGB2 = imresize(imycc, [224 224]);
baseFileName = sprintf('1-%d.png', k); % e.g. "1.png"
fullFileName = fullfile(Resizefolder, baseFileName); % No need to worry about slashes now!
imwrite(RGB2, fullFileName);
end
end
function Rename(Resizefolder,RenameFolder)
S = dir(fullfile(Resizefolder,'*.png'));
for k = 1:numel(S)
    F = fullfile(Resizefolder,S(k).name);
    imycc = imread(F);
   
baseFileName = sprintf('1-%d.png', k); % e.g. "1.png"
fullFileName = fullfile(RenameFolder, baseFileName); % No need to worry about slashes now!
imwrite(imycc, fullFileName);
end
end
function ColorSpace(RenameFolder,TestFolder)
S = dir(fullfile(RenameFolder,'*.png'));
for k = 1:numel(S)
    F = fullfile(RenameFolder,S(k).name);
    imycc = imread(F);
    XYZ = rgb2xyz(imycc);
 
baseFileName = sprintf('1-%d.png', k); % e.g. "1.png"
fullFileName = fullfile(TestFolder, baseFileName); % No need to worry about slashes now!
imwrite(XYZ, fullFileName);
end
end
function segmention(x)
bmean=0.1753;
rmean=0.1205;
brcov=[0.0022 0.0017 ; 0.0017    0.0014];

folder = imread(x);
BW = im2bw(folder,0.9)


s = regionprops(BW,BW,{'PixelValues','BoundingBox'});
numObj = numel(s);
for k = 1 : numObj
    s(k).PixelValues = std(double(s(k).PixelValues));
end


sStd = [s.PixelValues];
lowStd = find(sStd < 50 );


i=0;
for k = 1 : length(lowStd)
       rectangle('Position',s(lowStd(k)).BoundingBox,'EdgeColor','y');
       FishCrop=imcrop(folder,s(lowStd(k)).BoundingBox);
       baseFileName = sprintf('%d.png', i); 
       fullFileName = fullfile('C:\Users\ASUS\Desktop\GP\Image Classification by cnn\after segmention', baseFileName); 
       imwrite(FishCrop, fullFileName);
       i=i+1; 
end
xyz1 = rgb2xyz(folder);


res = zeros(size(folder,1),size(folder,2));
for b1 = 1:size(xyz1,1)
    for r1 = 1:size(xyz1,2)
        x1 = [double((xyz1(b1,r1,1)) - double(bmean)); 
            (double(xyz1(b1,r1,2))- double(rmean))];
 res(b1,r1) = exp(-0.1*x1'*inv(brcov)*x1);    
    end
end



T = adaptthresh(res,0.4);
BW = imbinarize(res,T);



s = regionprops(BW,BW,{'PixelValues','BoundingBox'});
numObj = numel(s);
for k = 1 : numObj
    s(k).PixelValues = std(double(s(k).PixelValues));
end


sStd = [s.PixelValues];
lowStd = find(sStd < 50);


i=0;
for k = 1 : length(lowStd)
    if s(lowStd(k)).BoundingBox > 5
      rectangle('Position',s(lowStd(k)).BoundingBox,'EdgeColor','y');
      FishCrop=imcrop(folder,s(lowStd(k)).BoundingBox);
      baseFileName = sprintf('%d.png', i); 
      fullFileName = fullfile('C:\Users\ASUS\Desktop\GP\Image Classification by cnn\after segmention', baseFileName); 
      imwrite(FishCrop, fullFileName);
      i=i+1;
    end

end


end
function TrainDisease(rootFolder,categories,TestFolder)

label='';
net = alexnet;

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
imageSize=[227,227,3];
D = 'C:\Users\ASUS\Desktop\Image Classification by cnn\Fish Diseases\Test';
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
if(strcmp(string(label),'1-XYZ-EUS')&&isequal(string(newChr),'2'))
      
accuracy=accuracy+1;
conn = database('phpmyadmin3','root','');
            c = date;

 colnames = {'date','farmhardwareid','disease'};
    insert(conn,'disease',colnames,{c,"aaaa","aaaaa"})

elseif (strcmp(string(label),'2-XYZ-ICH')&&isequal(string(newChr),'3'))
   conn = database('phpmyadmin3','root','');
            c = date;

 colnames = {'date','farmhardwareid','disease'};
    insert(conn,'disease',colnames,{c,"aaaa","aaaaa"})

accuracy=accuracy+1;
elseif (strcmp(string(label),'3-XYZ-columnaris')&&isequal (string(newChr),'1'))
accuracy=accuracy+1 ;
conn = database('phpmyadmin3','root','');
            c = date;

 colnames = {'date','farmhardwareid','disease'};
    insert(conn,'disease',colnames,{c,"aaaa","aaaaa"})

elseif(strcmp(string(label),'4-XYZ-NormalFish')&&isequal(string(newChr),'4'))
accuracy=accuracy+1 ;

end
end
end
function videoframes(numFrames,bbox)

for ii = 1:size(numFrames,1)
for j=1:size(bbox,1)
  fnvideo(numFrames(ii,:),bbox(j,1:4));
end

end
end
function fnvideo(videoFrame,bbox)
	
    myVideo = VideoWriter('Myfile.avi');
    open(myVideo);
    clust_file=fopen('velocity.txt','w');
    videoFileReader = VideoReader('testvideo1.avi');
    AverageMatrix=[0];
    videoFrame= readFrame(videoFileReader);
for j=1:size(bbox,1)
objectImage = insertShape(videoFrame,'Rectangle',bbox(j,1:4),'Color','red');
%figure;
%imshow(objectImage);
%title('Red box shows object region');

bboxPoints = bbox2points(bbox(j,1:4));
 points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox(j,1:4));

% Display the detected points.
%figure, imshow(videoFrame), hold on, title('Detected features');
%plot(points);

pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, videoFrame);

oldPoints = points;
end


while hasFrame(videoFileReader)

	% read video frame
	    videoFrame= readFrame(videoFileReader);

	% process fr
    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    
    if size(visiblePoints, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
                
        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);
         frameRate = videoFileReader.FrameRate; % frame/second
        scale = 1/320; % m/pixel
        M = mean(oldPoints);
        MM = mean(points);
    if ~isempty(M)
        % Calculate velocity (pixels/frame)
        vel_pix = sqrt(sum((MM-M).^2,2));
        vel = vel_pix * frameRate * scale; % pixels/frame * frame/seconds * meter/pixels
    else
        vel_pix = 0;
        vel = 0;
    end
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
            'LineWidth', 2);
                
        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints, 'o', ...
            'Color', 'red');       
        
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);  
         I=vel*100;
         
         
       AverageMatrix=[I;AverageMatrix];
       for ii = 1:size(AverageMatrix,1)
       fprintf(clust_file,'%6f \n',AverageMatrix(ii,:)); 
       end


        % Reset the points

        M=MM;
    end
    
   writeVideo(myVideo, videoFrame);
	pause(1/videoFileReader.FrameRate);
    
end 

fclose(clust_file);

 image=plot(AverageMatrix,'-mo',...
    'LineWidth',2,...
    'MarkerEdgeColor','k',...
    'MarkerFaceColor',[.49 1 .63],...
    'MarkerSize',10)
  xlabel('Velocity ');

saveas(image,"Velocity Tracking.png")



  
end