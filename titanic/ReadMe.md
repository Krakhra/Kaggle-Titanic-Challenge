Kaggle Titanic: Machine Learning from Disaster Competition 

Competition Description:

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

Learning Opportunities:

First start to finish machine learning project. Used the titanic datasets to predict which passengers where likely to survive with 80% accuracy. First thing I learned was how to preprocess data. Some features were grouped like age and fares, and unimportant features were dropped like ticket and name. 

I also learned how encoding data will make data more flexible for different algorithms. I used the labelencoder from scikit learn to convert each unique string into a number. 

Finally, I used support vector machine to classify the dataset. I chose to use a Nonlinear SVM to classify the dataset since the data is clustered in different spots. This is probably due to the different factors to lead to survival like cabin location age and sex. 
