[Go to main page](../README.md)

## Introduction to Machine Learning and Data mining

This is an Introduction to Machine Learning in R, in which you'll learn the basics of unsupervised learning for pattern recognition and supervised learning for prediction. At the end of this workshop, we hope that you will:
- appreciate the importance of performing exploratory data analysis (or EDA) before starting to model your data.
- understand the basics of unsupervised learning and know the examples of principal component analysis (PCA) and k-means clustering.
- understand the basics of supervised learning for prediction and the differences between classification and regression.
- understand modern machine learning techniques and principles, such as test train split, k-fold cross validation and regularization.
- be able to write code to implement the above techniques and methodologies using `R`, `caret` and `glmnet`.

We will not be focusing on the mathematical foundation for each of the methods and approaches we'll be discussing. There are many resources that can provide this context, but for the purposes of this workshop we believe that they are beyond the scope.

### Machine Learning basic concepts

Machine Learning (ML) is a subset of Artificial Intelligence (AI) in the field of computer science that often uses statistical techniques to give computers the ability to “learn” (i.e., progressively improve performance on a specific task) with data, without being explicitly programmed.

Machine Learning is often closely related, if not used as an alternate term, to fields like Data Mining (the process of discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems), Pattern Recognition, Statistical Inference or Statistical Learning. All these areas often employ the same methods and perhaps the name changes based on the practitioner’s expertise or the application domain.


### Taxonomy of ML and examples of algorithms

The main ML tasks are typically classified into two broad categories, depending on whether there is "feedback" or a "teacher" available to the learning system or not.

- **Supervised Learning**: The system is presented with example inputs and their desired outputs provided by the “teacher” and the goal of the machine learning algorithm is to create a mapping from the inputs to the outputs. The mapping can be thought of as a function that if it is given as an input one of the training samples it should output the desired value.
- **Unsupervised Learning**: In the unsupervised learning case, the machine learning algorithm is not given any examples of desired output, and is left on its own to find structure in its input.

The main machine learning tasks are separated based on what the system tries to accomplish in the end:
- **Dimensionality Reduction**: simplifies inputs by mapping them into a lower-dimensional space. Topic modeling is a related problem, where a program is given a list of human language documents and is tasked with finding out which documents cover similar topics.
- **Clustering**: a set of inputs is to be divided into groups. Unlike in classification, the groups
are not known beforehand, making this typically an unsupervised task.
- **Classification**: inputs are divided into two or more classes, and the learner must produce a model that assigns unseen inputs to one or more (multi-label classification) of these classes. This is typically tackled in a supervised manner. Identification of patient vs cases is an example of classification, where the inputs are gene expression and/or clinical profiles and the classes are "patient" and "healthy".
- **Regression**: also a supervised problem, the outputs are continuous rather than discrete.
- **Association Rules learning** (or dependency modelling): Searches for relationships between inputs. For example, a supermarket might gather data on customer purchasing habits. Using association rule learning, the supermarket can determine which products are frequently bought together and use this information for marketing purposes. This is sometimes referred to as market basket analysis.

### Overview of Deep learning

Deep learning is a recent trend in machine learning that models highly non-linear representations of data. In the past years, deep learning has gained a tremendous momentum and prevalence for a variety of applications. Among these are image and speech recognition, driverless cars, natural language processing and many more. Interestingly, the majority of mathematical concepts for deep learning have been known for decades. However, it is only through several recent developments that the full potential of deep learning has been unleashed. The success of deep learning has led to a wide range of frameworks and libraries for various programming languages. Examples include `Caffee`, `Theano`, `Torch` and `TensorFlow`, amongst others.

The R programming language has gained considerable popularity among statisticians and data miners for its ease-of-use, as well as its sophisticated visualizations and analyses. With the advent of the deep learning era, the support for deep learning in R has grown ever since, with an increasing number of packages becoming available. This section presents an overview on deep learning in R as provided by the following packages: `MXNetR`, `darch`, `deepnet`, `H2O` and `deepr`. It's important noting that the underlying learning algorithms greatly vary from one package to another. As such, the following table shows a list of the available methods/architectures in each of the packages.

| PACKAGE | AVAILABLE ARCHITECTURES OF NEURAL NETWORKS |
|---------|--------------------------------------------|
| MXNetR | Feed-forward neural network, convolutional neural network (CNN) |
| darch | Restricted Boltzmann machine, deep belief network |
| deepnet | Feed-forward neural network, restricted Boltzmann machine, deep belief network, stacked autoencoders |
| H2O | Feed-forward neural network, deep autoencoders |
| deepr | Simplify some functions from H2O and deepnet packages |

### Slides / material

The slides for this section are available [here](https://raw.githubusercontent.com/fpsom/2021-06-ml-elixir-fr/main/static/files/IntroML_ELIXIR-FR_Jun-2021.pdf)
