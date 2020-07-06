

D = 'C:\Users\Owner\Desktop\Salma';
S = dir(fullfile(D,'*.png'));

folder = 'D:\Images\salma';
for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    imycc = imread(F);
    

   
 
 
   
   
baseFileName = sprintf('1-%d.png', k); % e.g. "1.png"
fullFileName = fullfile(folder, baseFileName); % No need to worry about slashes now!
imwrite(imycc, fullFileName);
end





