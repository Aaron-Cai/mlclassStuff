clear all;
img = imread('test.png');
%300*300*3
imgGray = rgb2gray(img);
%%0~255
imgGray = double(imgGray);
imshow(imgGray);
imgGray = (imgGray - 255.0/2)/(255.0/2);
figure;
imagesc(imgGray);

imgGray = imresize(imgGray,[20 20]);
imagesc(imgGray,[-1 1]);