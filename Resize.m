



D = 'C:\Users\Owner\Desktop\resizemyimg.com 22-01-2020 13_06_23';
S = dir(fullfile(D,'*.png'));

folder = 'C:\Users\Owner\Desktop\ICH';
for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    imycc = imread(F);
    
  

RGB2 = imresize(imycc, [224 224]);
disp(size(RGB2));
    

   
 
 
   
   
baseFileName = sprintf('1-%d.png', k); % e.g. "1.png"
fullFileName = fullfile(folder, baseFileName); % No need to worry about slashes now!
imwrite(RGB2, fullFileName);
end


