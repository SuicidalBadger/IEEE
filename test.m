% read the test matrix in the same way we read the training matrix
N = dlmread('C:\Users\conno\Downloads\ex6DataPrepared\test-features.txt', ' ');
spmatrix = sparse(N(:,1), N(:,2), N(:,3));
test_matrix = full(spmatrix);

% Store the number of test documents and the size of the dictionary
numTestDocs = size(test_matrix, 1);
numTokens = size(test_matrix, 2);

% The output vector is a vector that will store the spam/nonspam prediction
% for the documents in our test set.
output = zeros(numTestDocs, 1);

% Calculate log p(x|y=1) + log p(y=1)
% and log p(x|y=0) + log p(y=0)
log_a = test_matrix*(log(prob_tokens_spam))' + log(prob_spam);
log_b = test_matrix*(log(prob_tokens_nonspam))'+ log(1 - prob_spam);
output = log_a > log_b;

% Read the correct labels of the test set
test_labels = dlmread('C:\Users\conno\Downloads\ex6DataPrepared\test-labels.txt');

% Compute the error on the test set
% A document is misclassified if it's predicted label is different from
% the actual label, so count the number of 1's from an exclusive "or"
numdocs_wrong = sum(xor(output, test_labels));

% Calculate the confusion matrix
TP = sum(output == 1 & test_labels == 1);
FP = sum(output == 1 & test_labels == 0);
TN = sum(output == 0 & test_labels == 0);
FN = sum(output == 0 & test_labels == 1);

% Convert the test_labels and output variables to double
test_labels = double(test_labels);
output = double(output);

% Compute the confusion matrix
C = confusionmat(test_labels, output);

% Display the confusion matrix
disp('Confusion Matrix:');
disp(C);


% Calculate Precision, Recall, and F1 Score
Precision = TP / (TP + FP);
Recall = TP / (TP + FN);
F1Score = 2 * Precision * Recall / (Precision + Recall);

%Print out statistics on the test set
fraction_wrong = numdocs_wrong/numTestDocs;
accuracy = 1 - fraction_wrong;

disp(['Number of misclassified documents: ' num2str(numdocs_wrong)]);
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Precision: ' num2str(Precision)]);
disp(['Recall: ' num2str(Recall)]);
disp(['F1 Score: ' num2str(F1Score)]);