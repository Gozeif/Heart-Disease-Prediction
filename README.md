# Brief Model Explanations
## Logistic Regression
### How it works:
finds a linear boundary in feature space that best separates the two classes.
The output is a probability, which is then thresholded at 0.5 to produce a class label.

### Hyperparameters we vary:

#### C — the inverse of regularisation strength.
A small C means strong regularisation,
which shrinks coefficients toward zero and prevents overfitting.
A large C gives the model more freedom to fit the training data.

#### solver — the optimisation algorithm used to find the best coefficients.
'lbfgs' and 'liblinear' are both good for small datasets like this one.
## Decision Tree
### How it works:
recursively splits the data on the feature/threshold that best separates the classes,
forming a tree of if/else rules.
Very interpretable — you can literally follow the path from root to
leaf to understand any individual prediction.

### Hyperparameters we vary:

#### max_depth — how deep the tree is allowed to grow.
A shallow tree underfits (misses patterns);
a very deep tree overfits (memorises training data).

#### min_samples_split — the minimum number of samples a node must have before it can be split further.
Higher values make the tree more conservative and less likely to create splits that only explain
one or two training examples.
## Random Forest
### How it works:
builds many decision trees, each trained on a random subset of rows (bagging)
and a random subset of features at each split.
The final prediction is a majority vote across all trees.
This reduces the variance that single decision trees are prone to.

### Hyperparameters we vary:

#### n_estimators — the number of trees in the forest.
More trees generally means better performance up to a point, after which returns diminish.
More trees also means longer training time.

#### max_depth — same concept as in the Decision Tree
but now applied to each individual tree in the forest.
Shallower trees mean more bias but less variance — the ensemble compensates for the bias.
## XGBoost
### How it works:
like Random Forest, XGBoost builds many trees
but instead of building them in parallel independently,
it builds them sequentially.
Each new tree focuses on correcting the errors the previous trees made.
This is called gradient boosting,
and it tends to produce very accurate results on structured/tabular data.

### Hyperparameters we vary:

#### n_estimators — number of boosting rounds (trees).
Unlike Random Forest, more rounds can lead to
overfitting, so this needs to be tuned carefully.

#### learning_rate — how much each new tree's contribution
is scaled down before being added to the ensemble.
A lower rate means each tree contributes less,
requiring more trees to reach the same performance
— but typically generalises better.

#### max_depth — depth of each individual tree. XGBoost
trees are typically kept shallow (3-6) because
the boosting process itself adds complexity.
## K-NN
### How it works:

no training in the traditional sense.
Instead, when predicting a new patient, KNN finds
the k most similar patients in the training set
(using distance in feature space) and takes a
majority vote of their labels. It's entirely
instance-based — the "model" is just the training
data itself.

This is the primary reason StandardScaler was applied
in preprocessing. Without scaling, Age (range ~50)
and Cholesterol (range ~200+) would completely
dominate the distance calculation, drowning out
binary features like Gender or Exercise Angina.
Scaling puts all features on equal footing.

### Hyperparameters we vary:

#### n_neighbors (k) — how many neighbours vote on the
prediction. Small k = sensitive to noise (overfits);
large k = smoother boundary (underfits). Finding
the right k is the central tuning challenge in KNN.

#### metric — the distance function used to measure
similarity. 'euclidean' is straight-line distance.
'manhattan' sums absolute differences dimension by
dimension. On high-dimensional data, manhattan
often generalises better than euclidean.
