% read the iris dataset:
data = readtable('iris.csv');
%Use  table to create a testing dataset (25% of the data) and a training dataset (75% of the data):
nTest = round(0.25 * size(data,1))
% re-seed Matlab's random number generator:
rng(1)
% shuffle the data and create a testing dataset and a training dataset:
data_shuffled = data(randperm(size(data,1)), :);
data_test = data_shuffled(1:1:nTest, :);
size(data_test)
data_train = data_shuffled(nTest+1:1:end, :);
size(data_train)
% separate the examples and the labels for the testing dataset:
test_labels = categorical(data_test{:,'species'});
test_examples = data_test;
test_examples(:,'species') = [];
% separate the examples and the labels for the training dataset:
train_labels = categorical(data_train{:,'species'});
train_examples = data_train;
train_examples(:,'species') = [];

% train our own Decision Tree model:
my_m = mytree.fit(train_examples, train_labels);
% use our model to classify the testing dataset:
my_predictions = mytree.predict(my_m, test_examples);

% output a confusion matrix:
[c,order] = confusionmat(test_labels, my_predictions)
% calculate the overall classification accuracy:
my_p = sum(diag(c)) / sum(c(1:1:end))