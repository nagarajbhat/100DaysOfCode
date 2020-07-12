I have decided to take part in the #100daysofcode challenge.

This is to record my journey and to learn and create awesome stuff.

So let us take a look at what it involves - 

- I will code for at least an hour every day.

- I will be posting every day under the tag #100daysofcode here and on my twitter.

- It is ok to miss a day (add additional days in the end), try not to miss two in a row.

- Most of my code will be related to python, Machine learning, and Deep Learning.

- **I will be uploading my code on Github and keep a log of it in this repository**. 

- If the project is interesting/useful I will write a blog post on my website.


Also, I would encourage other people to take a look at this challenge too and participate in it.
Check out the [original blog post](https://www.freecodecamp.org/news/join-the-100daysofcode-556ddb4579e4/) for more information on this challenge.

## My links - 

- [linkedin](https://www.linkedin.com/in/nagarajbhat12/)
- [twitter](https://twitter.com/nagarajbhat92)
- [my website](https://www.nagarajbhat.com)

#100daysofcode #coding #machine learning #100daysofmlcode


## Log -

### Day 1 - Pandas an small intro guide (olympics data)

Created a small intro guide for python's pandas framework. 

What is covered ? - 
- Read and select data
- Summary statistics
- Group and sort data
- create dataframe
- Combine data (merge,concat)

![Pandas basic methods](https://github.com/nagarajbhat/100DaysOfCode/blob/master/Pandas/pandas_basic_methods.png)

my links -
- [code](https://github.com/nagarajbhat/100DaysOfCode/blob/master/Pandas/pandas_intro_guide_olympics_data.ipynb)
- [blog post](https://www.nagarajbhat.com/post/picture-pandas-little-guide/)

other reources -
- [Pandas kaggle course](https://www.kaggle.com/learn/pandas)
- [pandas official user guide](https://pandas.pydata.org/docs/user_guide/index.html)


### Day 2 - Distance metrics 

- Explored different types of Distance metric - Euclidean,manhattan, hamming,minkowski.
- Distance metrics are used in algorithms such as KKN, K-means.
- Coded my own Distance metrics from scratch (improvements to be made) - [Code link](https://github.com/nagarajbhat/100DaysOfCode/blob/master/Algorithms_from_scratch/distance_metrics_from_scratch.ipynb)
- sklearn and scipy both provide tools to calculate distance metrics (refer code).

Usually, Euclidean is used as the distance metric, although the paper below [2] explains why Manhattan is preferable over euclidean for high dimentional data.
Obviously there's a lot more to cover regarding distance metrics and preferences.

![distance metrics notes](https://github.com/nagarajbhat/100DaysOfCode/blob/master/Algorithms_from_scratch/distance_metrics_notes.jpg)

Additional resources - 
- [1] [Sklearn Distance metrics - Docs](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html)
- [2] [On the Surprising Behavior of Distance Metrics
in High Dimensional Space - Paper](https://bib.dbvis.de/uploadedFiles/155.pdf)
- [3] [Different Types of Distance Metrics used in Machine Learning - blog](https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7)

## Day 3,4 - KNN for classification (of glass types)

- Explained KNN algorithm, used it to classify glass types.
- covered scatterplot,pairplot, correlation matrix (heatmap)
- Applied feature scaling
- Applied KNN to classify
- Optimization -
  - Distance matrix
  - finding the best K value.

Check out the [code here](https://github.com/nagarajbhat/100DaysOfCode/blob/master/classification/KNN_for_classification_of_glass_types.ipynb) , and my blog post on this topic [here](https://www.nagarajbhat.com/post/knn-for-classification/)

![KNN board](https://github.com/nagarajbhat/100DaysOfCode/blob/master/classification/KNN_board.PNG)

Other useful resources - 
- [K Nearest Neighbour Easily Explained with Implementation by Krish Naik - video](https://www.youtube.com/watch?v=wTF6vzS9fy4)
- [KNN by sentdex -video](https://www.youtube.com/watch?v=1i0zu9jHN6U)
- [KNN sklearn - docs ](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [Complete guide to K nearest neighbours - python and R - blog](https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/)
- [Why scaling is required in KNN and K-Means - blog](https://medium.com/analytics-vidhya/why-is-scaling-required-in-knn-and-k-means-8129e4d88ed7)

## Day 5,6,7 - Completed Intermediate machine learning course on Kaggle, also my improved housing prices competition model to rank in the top 8%.

![intermediate ml course kaggle](https://github.com/nagarajbhat/100DaysOfCode/blob/master/Courses/certificates/Nagaraj%20bhat%20-%20Intermediate%20Machine%20Learning.png)

What I like about these series of course from Kaggle is the hands-on-practical learning approach. 
This one is well compiled by Alexis Cook.

It covers - handling missing values, categorical variables, pipelines, XGboost, cross-validation, and data leakage.

Key Takeaways - 

1. Applying Machine learning pipelines resolved much of my headache in dealing with messy workflows. This is also good for quick experimentation, and reproducibility.
2. XGBoost is a state of the art ML model dominating most of the Kaggle competitions nowadays (effective for categorical data). Techniques for hyperparameter tuning was useful.
3. Data leakage can be very subtle and gets easily ignored. Think of it like this - If you do not have access to a particular feature/predictor before making a prediction for new data, then that feature shouldn't be used in the model in the first place.

I have compiled the code structures for future reference's sake. (don't mind taking it down if requested).

[code link](https://github.com/nagarajbhat/100DaysOfCode/blob/master/Courses/Intermediate_ML_kaggle.ipynb)

other links - 
- [Housing price competition](https://www.kaggle.com/c/home-data-for-ml-course)
- [Intermediate machine learning course - Kaggle](https://www.kaggle.com/learn/intermediate-machine-learning)

## Day 8,9,10,11: Wrote a blog - "Predicting Titanic movie's character's survival chance using Machine learning". I Improved my titanic dataset model to achieve an accuracy of 80.382%  - currently ranking in the top 9% in Kaggle.

Although this is one of the most common datasets, I was curious to compare the movie character's survival with the actual prediction. I also covered how to improve model accuracy. 

check out the complete blog [here](https://www.nagarajbhat.com/post/predicting-titanic-survival/)

Disclaimer - This blog is not about whether Jack could fit on the door with Rose!

![Jack-rose-meme](https://www.nagarajbhat.com/post/predicting-titanic-survival/featured.jpg)

Apart from this, 
- I worked on feature engineering and selection techniques
- Learned about categorical encodings such as count encoding, target encoding, catboosting.
- Studied a bit about docker.
- Studied about lightGBM.

## Day 12-17 : Created a simple python flask app called Refresh Quotes, this displays a new quote every time you refresh!

It is currently deployed on Heroku, do check it out here - > [Refresh Quotes](https://refreshquotes.herokuapp.com/)  
Here's the [code repo](https://github.com/nagarajbhat/refreshquotes)

Features - 
- displays a new quote, author every time you refresh the page or press the button.
- Can look up the author on google with a click

![refresh quotes image](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/refreshquotes_desktop.PNG)

Learnings - 
- You can use mix-blend-mod , filter dropshadow, invert to differentiate text from background in CSS.
- Jinja template designer for internal logic in an HTML file. 
- Heroku deployment - using Procfile, GitHub integration, and Heroku CLI.

I may get back to this project for new features (possibly some NLP), and improving content.

Also, I thought it would be a good idea to refresh some of the basic programming concepts, So I started with OOPs(Object Oriented Programming) for python. Coded class methods and variables, inheritance,etc. - [link](https://github.com/nagarajbhat/100DaysOfCode/blob/master/Algorithms_DataStructures/OOPS_concepts.py)

## Day 18,19: Used pyspark for a multi-class Classification problem.
Used grid search to find the best values for the hyperparameters .
Use "isLargerBetter" method to determine if the metrics returned by evaluater needs to be maximised or minimised

## Day 20: I Created a Subscription form for my website content using Mailchimp.
If you are interested in Machine learning, python, statistics, 100DaysOfCode, and want to receive my posts on it along with some good learning resources that I refer to, do subscribe!

![subscribe](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/subscribe.png)

I know the popup can be irritating, so it coded it such that it will only appear when you are top or bottom part of my website, and disappears while you are reading the blog.


## Day 21 : I am recontinuing this challenge after a gap. Today I explored clustering for dimensionality reduction.

We know that clustering is an unsupervised learning technique. 
Here I explored how Kmeans clustering can be used as a preprocessing step for dimensionality reduction before applying a supervised learning technique to classify digits in MNIST handwritten digit dataset.

[code link](https://github.com/nagarajbhat/100DaysOfCode/blob/master/classification/MNIST_KmeansFordimnesionalityReduction.ipynb)

![KMeans mnist image](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/kmeans_mnist.PNG)

Applying logistic regression to classify the digits, we get an accuracy score of 94.8%. Using Kmeans as a preprocessing step in the pipeline to cluster instances together increased the accuracy to 97.5%. 

Note that even though we classify 10 digits, n_clusters = 10 is not likely to give the best result as there are several ways to write each digit. The best value for the number of clusters can be found using GridSearchCV. Here it was found to be 78.

Further, we can also use clustering for semi-supervised learning where some of the instances are unlabelled. Here, unlabelled instances can be assigned a label based on the cluster they belong to.

reference - Hands on ML with scikit,keras and tensoflow (Book)
 
 ## Day 22: Dealing with the curse of dimensionality
Data behaves differently in high dimensional space when compared to lower dimensions.
For example, As the number of dimensions increases the distance between any 2 random data points also increases, in high dimensional data, the dataset is sparse.
This means that new instances are very far away from the training data, making the predictions unreliable.

#DimensionalityReduction : 
Projection- In the real world, data points in high dimensions actually lie or are close to each other in a lower dimension subspace, thus they can be projected into a lower dimension. However, this may not be the best approach for data with twists (eg - swiss roll dataset)
![code_projection](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/code_projection.PNG)

Manifold learning - swiss roll is an example of a 2D manifold. Here the higher dimensional data points can be twisted into a lower-dimensional plane.
The drawback is that the decision boundary is not always simple.
![manifold](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/manifold_learning.PNG)

PCA - the point of Principal component analysis(PCA) is to project high dimensional data into a lower subspace in such a way that you preserve the variance. One method to do this is to pick the axis which minimizes the mean squared distance between the original dataset and its projected version.
Applying PCA to MNIST can result in compression of the image.
![PCA](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/PCA.PNG)

links - 
- [my code](https://github.com/nagarajbhat/100DaysOfCode/blob/master/dimensionality_reduction/dimensionality_reduction.ipynb)
- Hands on Machine learning with scikit-learn, keras and tensorflow [2nd edition] (Book)
- [scikit-learn unsupervised dimensionality reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html)

## Day 23,24: Here is a reason why considering just the accuracy metric for your ML model is not enough.
I am currently building a Dashboard for pokemon. In one of the tab, I predict the type of pokemon using features such as Attack, Defense, Speed, etc.

According to the dataset, there are 18 types of pokemon (Water, Dragon,etc.). So I ran an ML model to classify between any two types.

Scenario 1 - Bug vs Grass gives an accuracy of 86%
Scenario 2 - Water vs Ghost gives an accuracy of 90%. Does this mean that scenario 2 performs better?

The problem is that the dataset is imbalanced. In scenario 1, each class has an almost equal share. But in scenario 2, type Water takes more than 77% of the share, which means that the model will give high accuracy if it manages to classify Water-type properly (even if it classifies Ghost-type poorly).
![pokemon type](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/pokemon_type.PNG)

So consider other metrics such as precision, recall, F1-score to get a better perspective.
Precision - determines how much of what is classified as type "A" is actually "A".
Recall - determines How much is classified as type "A" out of all that could be classified as "A".

While the model does classify Water-type well, Its F1-score for Ghost is low. Whereas Scenario 1 gives reasonable precision, recall values for both the types.
![scenario 1](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/scenario1.PNG)
![scenario 2](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/scenario2.PNG)

## Day 25-29 : I built my first Plotly Dash app called "Dash pokemon". 

The app is deployed on Heroku and is live [here](http://dash-pokemon.herokuapp.com/) .
It is for visual analysis and pokemon type prediction and could be useful for pokemon related gaming.

Dash is a framework for building Dynamic dashboards, it's built on top of Flask and Reactjs.

![demo gif](https://github.com/nagarajbhat/dash-pokemon/blob/master/screenshots/demo_gif.gif)

![tab 2](https://github.com/nagarajbhat/dash-pokemon/blob/master/screenshots/tab2.PNG)

The app has 4 tabs -
- Tab 1 - Bar graph visual of best pokemon using Pokemon type and strength criteria.
- Tab 2 - Box plot for each pokemon type. Also finds the best and worst types.
- Tab 3 - Scatterplot, pairplot - Correlations between different features of pokemon.
- Tab 4 - Predicting a Pokemon's type using its features. There are options to play around with features as well as the targets (pokemon's types) to be classified.
  Model accuracy varies according to the options selected.

Apart from dash core components and HTML components, I have made use of DBC (Dash bootstrap components) for styling.

I enjoyed using Plotly dash, and have learned a fair bit about how it works.
Let me know if you want me to make a video/blog post on developing apps/dashboards using Dash.

Here's the [code link](https://github.com/nagarajbhat/dash-pokemon)

## Day 30: Used black(a lint tool) for auto-formatting the code.
Linting checks for programmatic as well as stylistic errors in your program.
Black is Opiniated,deterministic and consistent.

![image](https://github.com/nagarajbhat/100DaysOfCode/blob/master/images/black_format.png)

Why use it?
- Improves readibility
- uniformity across all projects
- faster code reviews

Usage?
>pip install black //installation
>black . --check //checks files that needs reformatting
>black file_name.py //formats the file

read more on it here - [psf/black github](https://github.com/psf/black)
