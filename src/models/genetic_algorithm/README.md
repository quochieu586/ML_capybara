# Text Classification Model using Genetic Algorithm

## 1. Overview

- Genetic Algorithm is a machine learning algorithm that iteratively reproduce the a new population of hypothese over a previous population and finally return the best. In this project, we combine *Genetic Algorithm* to *Neural Network* to which is designed to classify text data into predefined sentiment categories. The idea is that, instead of using gradient descent to update weights, we apply *Genetic Algorithm* strategy in which we crossover the weights between models.

## 2. Key Features

- Traditional genetic algorithm is quite challenging to put into practice since it requires representing hypothesis in term of bits string which is very hard for large model. For that case, in this project, we modify the Genetic Algorithm by representing model into list of float numbers instead.

- A population of neural network models are initilized in which each model is flatten into a fixed-size string of weights. For model evaulation, we use *accuracy* as *Fitness score*.

- The training process follows the principle of Genetic Algorithm as belows:

    + *Model selection*: distribution of each model in population is computed based on *Fitness score*, model with higher fitness score has higher probability. Then, 2 models are randomly sampled from that distribution.

    + *Crossover*: 2 childrens are created by crossover the weights between 2 models.

    + *Mutation*: some weights in each children is updated by adding random noises.

- Feature extraction: *Word embedding* with PCA (30 features ~ 60% variance after applying PCA)

## 3. Model Architecture

- Neural network model:

    + **Input Layer**: The input layer is the feature vector from TF-IDF representation of the text data.

    + **Hidden Layer**: Two fully connected hidden layers with sigmoid activations and batch normalization are used to capture complex patterns in the data.

    + **Output Layer**: The output layer uses a softmax activation function to predict the probability of each sentiment class (negative, neutral, or positive).

- Genetic Algorithm:

    + **Population**: The population is the set of models that are initialed and updated during training.

## 4. Hyperparameter Tuning

In our model, we consider tuning the following hyperparamters:

+ **POP_SIZE**: number of neural networks in population

+ **MUTATION_RATE**: rate of weights being mutated during mutation

+ **MUTATION_NOISE**: define how much noise is added to mutated weights

## 5. Results

We shown the results of model prediction on test set as follows:

+ **Accuracy**: 0.4046

+ **F1-score**: 0.2331

+ **AUC-ROC**:
    + Negative: 0.5075
    + Neural: 0.4082
    + Positive: 0.3980

### Comment on result
The genetic agorithm shows a poor performance since it predicts 1 label (neural) for every sample. A potential problem may be from *mutation* and *crossover* strategy. Difference from traditional Genetic Algorithm where the unit of string is a bit (hence, finite search space), with float number, it cover a wider (infinite) hypothesis search space. Other weaknesses are discussed in the next part.

## 6. Model strength

+ **Gradient-Free Optimization**: GA approach does not require backpropagation and computing gradient descent. Hence, it is good in case the loss function gradient is non-differentiable.

+ **Global Search Ability**: Normal neural network can be stuck in local optimal while GA does not. It explores a larger search space.

## 7. Model weakness

+ **Computationally expensive**: GA works on many neural network models (population). During training, forwarding data to compute fitness scores in many models makes this model slower than traditional neural network a lot. 

+ **Storage expensive**: Keeping many neural networks in population is very expensive. Hence, it is impossible to use Tf-idf that requires ~10,000 feature for best training. For those reason, we choose to implement a lighter feature which is word embedding with PCA which reduces the feature size to 30.

+ **No Fine-Tuning Capability**: GA does not performs incremental learning liked traditional neural network with gradient descent.

+ **Requires Careful Hyperparameter Selection**: Mutation rate, crossover rate, and population size must be tuned. Poor settings can lead to premature convergence or very slow evolution. Also, as shown in the notebook, this model depends strongly on the initilization point, but this was solved using hyperparameter tuning.

+ **No Guarantee of Convergence**: Unlike gradient descent (which follows a clear path based on loss gradients), GA randomly explores the space. Hence, it may never find an optimal solution in a reasonable amount of time. In our notebook, it shows that these models always convergence to the base model where it always predict a single label for every samples.
