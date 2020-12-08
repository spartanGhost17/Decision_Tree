classdef mytree
    methods(Static)
        %% Define and create a tree 
        %%
        %% return structure m
        function m = fit(train_examples, train_labels)
            %% unique number of nodes within the tree
			emptyNode.number = [];
            %% training example and training labels held within the node
            emptyNode.examples = [];
            emptyNode.labels = [];
            %% prediction on any class label held in the node
            emptyNode.prediction = [];
            %% numeric measure of impurity of class labels held, determine whether to split the node 
            emptyNode.impurityMeasure = [];
            %% store children node (if node is determined to be split and divide training data between children)
            emptyNode.children = {};
            %% store column number on which split will be performed
            emptyNode.splitFeature = [];
            %% store name of feature on which split will be performed
            emptyNode.splitFeatureName = [];
            %% store value of feature on which split will be performed
            emptyNode.splitValue = [];
            %% assign empty node field structure with an empty node (root node)
            m.emptyNode = emptyNode;
            %% copy empty node in (r) to create the root node of the tree
            r = emptyNode;
            %% initial number of unique nodes in tree before any split is performed
            r.number = 1;
            %% copy all training labels
            r.labels = train_labels;
            %% copy all training_examples
            r.examples = train_examples;
            %% find the most common class label which will be the most likely class for classification of data held in node in case no purity split is required
            r.prediction = mode(r.labels);
            %% set minimum number of examples node must contain before considering spliting node. copy value in min_parent_size field of structure (m)
            m.min_parent_size = 10;
            %% copy list of unique class labels in the unique_classes field on the structure (m) 
            m.unique_classes = unique(r.labels);
            %% aggregate all feature names (column titles) in training data 
            m.feature_names = train_examples.Properties.VariableNames;
            %% current number of nodes in the tree
			m.nodes = 1;
            %% copy total number of training examples used for model training store in field (N) of m structure
            m.N = size(train_examples,1);
            %% try spliting the tree using model (m) and root node (r). This action generates our tree
            %% determine if node is candidate for splitting
            m.tree = mytree.trySplit(m, r);

        end
        %% Verify if a node split is required i.e node contains more than 10 examples (which is over the minimum size in m.min_parent_size)  
        %% allows to reduce impurity of class labels in node.
        %% return node when tree can no longer be split 
        function node = trySplit(m, node)
            
            %% if node size is less than min node size (10), leave method (not enough training data to become parent node/reduce node impurity of class label in node)
            if size(node.examples, 1) < m.min_parent_size
				return
			end
            %% evalute measure of impurity of current node labels, copies value in node.impurityMeasure field
            node.impurityMeasure = mytree.weightedImpurity(m, node.labels);
            %% for every feature in our training example in the node 
            for i=1:size(node.examples,2)

				fprintf('evaluating possible splits on feature %d/%d\n', i, size(node.examples,2));
                %% sort table of examples in node based on current feature (row), output sorted table (ps) and original indicies array (n)
				%% sorting will facilitate spliting of table.
                [ps,n] = sortrows(node.examples,i);
                %% copy labels at original indicies (array n), labels will be ordered like training examples 
                ls = node.labels(n);
                %% array of biggest reductions in node impurity 
                biggest_reduction(i) = -Inf;
                %% array of indicies with biggest reductions in node impurity
                biggest_reduction_index(i) = -1;
                %% initialize biggest reduction value
                biggest_reduction_value(i) = NaN;
                %% for every row in ordered table (ps) -1, check if value for feature X at row(n,m) is equal to value at (n+1,m). If they are the same, 
                %% then we do not need to split on that value (prevents us from spliting the same value more than once).
                for j=1:(size(ps,1)-1)
                    if ps{j,i} == ps{j+1,i}
                        continue;
                    end
                    %% calculate weighted impurity reduction. if and only if impurity in parent node is greater than combined impurities of both possible children then probable split is possible 
                    this_reduction = node.impurityMeasure - (mytree.weightedImpurity(m, ls(1:j)) + mytree.weightedImpurity(m, ls((j+1):end)));
                    %% if node can be split
                    if this_reduction > biggest_reduction(i)
                        %% keep track of largest impurity reduction for current feature in biggest_reduction array
                        biggest_reduction(i) = this_reduction;
                        %% record index of feature value allowing the biggest reduction to impurity in parent node in biggest_reduction_index 
                        biggest_reduction_index(i) = j;
                    end
                end
				
            end
            %% find biggest reduction in biggest_reduction array, output max value (winning_reduction) and its index (winning_feature)
            [winning_reduction,winning_feature] = max(biggest_reduction);
            %% get index of feature value to split on
            winning_index = biggest_reduction_index(winning_feature);
            %% if winning reduction is 0 or less, leave the method, node can't be split because no impurity reduction was acheived
            if winning_reduction <= 0
                return
            %% if reduction in impurity was acheived proceed to splitting node   
            else
                %% sort rows of data in node based on winning feature
                [ps,n] = sortrows(node.examples,winning_feature);
                %% find labels associated with node data at indicies in original indicies array (n)
                ls = node.labels(n);
                %% store index of winning feature in field node.splitFeature on which node data will be split
                node.splitFeature = winning_feature;
                %% name of the feature (row) containing value we want to split on
                node.splitFeatureName = m.feature_names{winning_feature};
                %% ---- instead of winning value only, split value will be value exactly half way between winning value and value right after it(which could not reduce impurity)
                node.splitValue = (ps{winning_index,winning_feature} + ps{winning_index+1,winning_feature}) / 2;
                %% delete training examples from current (parent) node, they will move in children nodes
                node.examples = [];
                %% delete training labels from current (parent) node, they will move in children nodes
                node.labels = []; 
                %% delete predictions, only leaf nodes can be used to generate predictions in prediction phase.
                node.prediction = [];
                %% copy empty node structure in node field cell array
                node.children{1} = m.emptyNode;
                %% update number of nodes in tree
                m.nodes = m.nodes + 1; 
                node.children{1}.number = m.nodes;
                %% copy training examples and training labels up to winning index  
                node.children{1}.examples = ps(1:winning_index,:); 
                node.children{1}.labels = ls(1:winning_index);
                node.children{1}.prediction = mode(node.children{1}.labels);
                %% copy empty node structre 
                node.children{2} = m.emptyNode;
                %% update number of nodes in tree
                m.nodes = m.nodes + 1;                 
                node.children{2}.number = m.nodes;
                %% copy training examples and training labels after winning index
                node.children{2}.examples = ps((winning_index+1):end,:); 
                node.children{2}.labels = ls((winning_index+1):end);
                node.children{2}.prediction = mode(node.children{2}.labels);
                %% recursive call trySplit function on children nodes, this process will be repeated until leaf nodes (no more possible split) are reached
                node.children{1} = mytree.trySplit(m, node.children{1});
                node.children{2} = mytree.trySplit(m, node.children{2});
            end

        end
        %% calculate measure of current impurity within a node class labels
        %%
        %% return weighted Impurity (needed to determine split of node)
        function e = weightedImpurity(m, labels)
             
            %% probability of descending in current node, will be used to rescale each GDI scores.
            %% Divide number of labels in current node by total number of training data 
            weight = length(labels) / m.N;

            summ = 0;
            %% number of labels observed in this node
            obsInThisNode = length(labels);
            %% for every class label in data set
            for i=1:length(m.unique_classes)
                %% calculate proportion of class label for current class labels in node (number of label of class X divided by total number of labels in node)
				pc = length(labels(labels==m.unique_classes(i))) / obsInThisNode;
                %% add square of fractions of current class label in set
                summ = summ + (pc*pc);
            
            end
            %% calculate Gini's diversity index (impurity index current node), closer to zero the purer node is
            g = 1 - summ;
            %% will be needed to compare impurity between parent and his children
            e = weight * g;

        end
        %% predict labels for test examples 
        %% return an array of predicted labels
        function predictions = predict(m, test_examples)
            %% initialze prediction categorical array 
            predictions = categorical;
            %% for every rows of data in testing data 
            for i=1:size(test_examples,1)
                
				fprintf('classifying example %i/%i\n', i, size(test_examples,1));
                %% collect row of testing example at current index
                this_test_example = test_examples{i,:};
                %% predict label for current test example by descending the tree and copy value in this_prediction 
                this_prediction = mytree.predict_one(m, this_test_example);
                %% add prediction to predictions categorical array (predictions)
                predictions(end+1) = this_prediction;
            
			end
        end
        %% predict label of individual example by descending down the trained tree, comparing split value and split feature to corresponding feature and value
        %% in current training example
        %% return predicted label
        function prediction = predict_one(m, this_test_example)
            %% copy leaf node returned from descending the tree 
			node = mytree.descend_tree(m.tree, this_test_example);
            %% copy prediction label for reached leaf node (most common label held by data in that node)
            prediction = node.prediction;
        
		end
        %% descend the tree and return a leaf node
        function node = descend_tree(node, this_test_example)
            %% if leaf node was reached return node
			if isempty(node.children)
                return;
            else
                %% if spliting value is less than value in node, descend left (left side holds small values right side holds higher values)
                if this_test_example(node.splitFeature) < node.splitValue
                    %% recursively call descend_tree to move down the tree
                    node = mytree.descend_tree(node.children{1}, this_test_example);
                %% else descend right child (bigger values on right side nodes)    
                else
                    %% recursively call descend_tree to move down the tree
                    node = mytree.descend_tree(node.children{2}, this_test_example);
                end
            end
        
		end
        
        % describe a tree:
        function describeNode(node)
            
			if isempty(node.children)
                fprintf('Node %d; %s\n', node.number, node.prediction);
            else
                fprintf('Node %d; if %s <= %f then node %d else node %d\n', node.number, node.splitFeatureName, node.splitValue, node.children{1}.number, node.children{2}.number);
                mytree.describeNode(node.children{1});
                mytree.describeNode(node.children{2});        
            end
        
		end
		
    end
end