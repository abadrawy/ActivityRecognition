# Activity Recognition Using Neural Networks and SVMs
Built two classifiers, a Nerual network classifier and an SVM classifer, that determine the physical activity of a human based on the readings of multiple smartphone sensors.

## Dataset
The Dataset used to tain the models, is the Human Activity Recognition dataset, collected by Genova University, 
and available via the University of California Irvine
[Machine learning repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).


## Neural Network Classifier
To find the optimal classifier for the neural network, 2 parameters were tuned, the alpha and the hidden_layer_sizes, 
using cross-k validation (3 fold was used) and plotting the curve.

First, cross validation with respect to hidden_layer_sizes was applied, having the maximum number of hidden layers as 2, the range values for the possible number of nodes in any of the 2 layers is [300,500,400,600,700], thus the range of all possible values is
[(300,), (400,), (500,), (600,), (700,), (300, 300),(300, 400), (300, 500), (300, 600), (300, 700), (400, 300), (400, 400), (400, 500), (400, 600), (400, 700), (500, 300), (500, 400), (500, 500), (500, 600), (500, 700), (600, 300), (600, 400), (600, 500), (600, 600), (600, 700), (700, 300), (700, 400),(700, 500), (700, 600), (700, 700)]
The best performing hidden_layer_sizes parameter is the 5th tuple as observed from the plot, which is (700,). Which strikes a balance between the number of nodes and number of layers.

![alt text](https://github.com/abadrawy/ActivityRecognition/blob/master/images/MLP_Hidden_layer_sizes.png)


Second, cross validation with respect to the alpha was applied, which is the regularization parameter, it controls the weights as to prevent classifier from over-fitting (if it was a high value) by encouraging smaller weights, prevents classifier from under-fitting by encouraging larger weights (if it was a low value).

The alpha ranges used = [0.00001, 0.001, 0.1, 1, 10]

The best performing alpha as seen from the plot is 10^0 which is 1.

![alt text](https://github.com/abadrawy/ActivityRecognition/blob/master/images/MLP_alpha.png)


## SVM Classifier
To find the best performing multiclass SVM, the parameters, gamma and C, were tuned using cross validation as well.

First, cross validation with respect to gamma was applied, which is the parameter of the kernel, and by default is the rbf(radial basis function), if gamma is large then variance is small and it implies that the support vector does not have much influence, while if it was small then it means the support vector will have more influence.

The gamma ranges used = [1, 0.1, 0.01, 0.001, 0.0001]

The optimal gamma is 10^-3 which is 0.001 as observed from the plot.

![alt text](https://github.com/abadrawy/ActivityRecognition/blob/master/images/SVM_gamma.png)

Second, we applied cross validation with respect to C, the C is the penalty parameter in SVM, it controls how it is tolerant to miss classified instances, it was introduced to achieve the soft margin, the smaller the C, the more tolerant to mistakes and the larger it is, the stricter the classifier is.

The C ranges used = [0.1, 1, 10, 100, 1000]

The optimal C value is 1000 (10^3) as observed from the validation curve plot below.

![alt text](https://github.com/abadrawy/ActivityRecognition/blob/master/images/SVM_C.png)




## The difference in performance between the two classifiers

The classifiers performance was compared in terms of training time and prediction accuracy.

Generally, SVM performs and yields better results then Neural Networks as they are guaranteed to reach the global minima and they have less hyper parameters, so the gird search space is much less.

#### Training Time:

The training time of SVM is much less than that of MLP as it’s learning strategy is faster and it does not have have to update a lot of weights each iteration.

SVM: 3.499 S

MLP: 15.584 S

#### Prediction Accuracy:

The prediction of the SVM is slightly more accurate in this case, and this may be because in SVM, the decision boundary is explicitly decided directly by the training data while in MLP it is decided by being adjusted based on the sum of squared error.

SVM: 95.9%

MLP: 95.3%


## Applying the PCA technique
By Varing the number of principal components to [5, 50, 200, 500]

PCA reduces the dimensions of data to the specified dimension, and the reduced data is represented based on it’s variance.

We could see that reducing the dimensions to just 5, decreased the performance significantly, and this is because 5 dimensions can not express the data properly and a lot of information was lost. We could also see that when we increased the dimensions to 50, it relatively increased, but still the result was still low when compared to that of the optimal classifier in both cases. When we increased the dimensions to 200, the results became nearly equal to that of the optimal classifier, and that means that 200 dimensions are enough to represent the data, almost just as well as the original, finally when we increased the dimensions to 500, In the MLP classifier the performance became equal to that of optimal classifier, but in the SVM it outperformed the optimal classifier, and this may be because the PCA helped the SVM concentrate on the most important features and thus learn better.

## Libriaries

* scikit-learn
* numpy
* pandas
* matplotlib
