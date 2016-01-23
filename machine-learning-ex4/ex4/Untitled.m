clear all;
%%=======================Load raw data and do preprocessing===============
tic; %start time recording
img = imread('test1.png');
img = rgb2gray(img);
img = double(img); %convert int to double
[row,col]=size(img);
sample = img(:)';     %unroll the 2d matrix to 1d row vector
%normalize data
for i = 1 : row*col
    sample(i) = (sample(i)-256/2)/128.0;
end
sampleY = 2; %the label of the sample

%%=======================Set parameters for neural network==============
input_layer_size = row * col; % we got row * col pixels
hidden_layer_size = 25; % 25 hidden units
num_labels = 10;        % 10 labels, from 1 to 10

%%========================random initialize weights =====================
ini_weights1 = rand(hidden_layer_size,input_layer_size + 1);
ini_weights2 = rand(num_labels, hidden_layer_size + 1);
ini_weights = [ini_weights1(:) ; ini_weights2(:)];

%% ======================Training NN ===================================
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, sample, sampleY, lambda);
options = optimset('MaxIter', 100);
[weights, cost] = fmincg(costFunction, ini_weights, options);

weights1 = reshape(weights(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

weights2 = reshape(weights((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%%======================Visualize weights===============================
displayData(weights1(:,2:end));
toc;

