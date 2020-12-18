# TCS_iON_RIO_AUTOMATEsentimentAnalysis
Automate Sentiment Analysis of Textual Comments and Feedback
(Artificial Intelligence)
Introduction and Importing Data:
IMDB movie reviews dataset is used for this study. The dataset contains 50,000 reviews — 25,000 positive and 25,000 negative reviews. An example of a review can be seen in Fig 1, where a user gave a 10/10 rating and a written review for the Oscar-winning movie Parasite (2020). The number of stars would be a good proxy for sentiment classification. For example, we could pre-assign the following:
At least 7 out of 10 stars => positive (label=1)
At most 4 out of 10 stars => negative (label=0)
 
Figure 1
Here is a preview of how the dataset looks like:
 
Figure 2
EXPLORATORY TEXT ANALYSIS:

In order to proceed with exploratory text analysis, firstly the data has been separated into training and test set, and then the data is split into sub-sets.
 


No. of Strings Present:
print(f"Number of strings: {len(splits)}")
print(f"Number of unique strings: {len(set(splits))}")
 
Most common strings:
freq_splits = FreqDist(splits)
print(f"***** 10 most common strings ***** \n{freq_splits.most_common(10)}", "\n")
 








Frequency of occurrence of html tags:
The example html tag: ‘<br /><br />’ would have been split into three strings: ‘<br’, ‘/><br’ and ‘/>’ when we split the data based on white space. On side note, the <br> tag seems to be used to break lines.
def summarise(pattern, strings, freq):
"""Summarise strings matching a pattern."""
compiled_pattern = re.compile(pattern)
matches = [s for s in strings if compiled_pattern.search(s)]
# Print volume and proportion of matches
print("{} strings, that is {:.2%} of total".format(len(matches), len(matches)/ len(strings)))
# Create list of tuples containing matches and their frequency
output = [(s, freq[s]) for s in set(matches)]
output.sort(key=lambda x:x[1], reverse=True)
return output
# Find strings possibly containing html tag
summarise(r"/?>?w*<|/>", splits, freq_splits)
 
Frequency of Punctuation Marks:
summarise(r"\w+[_!&/)(<\|}{\[\]]\w+", splits, freq_splits)
 



Most Frequent Stop-words:
stop_words = stopwords.words("english")
print(f"There are {len(stop_words)} stopwords.\n")
print(stop_words)

 

This Exploratory Text analysis will help to get an idea regarding what steps to proceed with in order to get an accurate sentiment prediction while creating or advancing with models.
In this Project report the performance of Logistic Regression and Naïve Bayes Algorithm are been accessed in determining the sentiment of a review.






LOGISTIC REGRESSION MODEL ANALYSIS AND PERFORMANCE:
For training, our feature matrix is greatly sparse, meaning there is a lot of zeros in this matrix as there are 25,000 rows and approximately 75,000 columns. So, what we’re going to do is find the weight of each feature and multiply them with their corresponding TD-IDF values; sum all the values, pass it through a sigmoid activation function, and that’s how we end up with the Logistic Regression model. 
The advantages of applying a Logistic function in this case: 
•	The model handles sparse matrices really well, and 
•	The weights can be interpreted as a probability for the sentiments.
Model: Logistic regression
•	𝑝(𝑦=1|𝑥)=𝜎(𝑤𝑇𝑥)p(y=1|x)=σ(wTx)
•	Linear classification model
•	Can handle sparse data
•	Fast to train
•	Weights can be interpreted
 
Figure 3










Transforming Documents into Feature Vectors
Below, we will call the fit_transform method on CountVectorizer. This will construct the vocabulary of the bag-of-words model and transform the sample sentence below into a sparse feature vector.
 
Figure 4
Raw term frequencies: tf (t,d)—the number of times a term t occurs in a document d.
Word relevancy using term frequency-inverse document frequency
 
where  𝑛d  is the total number of documents, and df(d, t) is the number of documents d that contain the term t.
 
Figure 5


The equations for the Idf and Tf-Idf that are implemented in scikit-learn are:
 
The Tf-Idf equation that is implemented in scikit-learn is as follows:
 
 
Example:
 
Now in order to calculate the Tf-Idf, we simply need to add 1 to the inverse document frequency and multiply it by the term frequency:
 
Calculation of Tf-Idf of the term “is”: 
 
 
  



Data Preparation:
We will define a helper function to pre-process the text data as our text of reviews can contain special characters — html tags, emojis — that we would want to account for when training our model.
The special characters, emoji’s and unwanted garbage values are removed throughout the string and added at the end of the string.
 
As seen above, after applying the defined preprocessor function, the sample sentence was stripped of special characters; the emojis were also moved to the end of the sentence. This is so that our model can make use of the sequence of the text and also ascertain the sentiment of the emojis at the end of the sentences.

Tokenisation of Documents
The data is represented as a collection of words or tokens; and also be performing word-level pre-processing tasks such as stemming. To achieve this, we will utilize the natural language toolkit, or nltk.
Stemming is a technic that reduces the inflectional forms, and sometimes derivationally related forms, of a common word to a base form. For example, the words ‘organizer’ and ‘organizing’ stems from the base word ‘organize’. So, stemming is conventionally referred to as a crude heuristic process that ‘chops’ off the ends of words, in the hope of achieving the goal correctly most of the time — this often includes the removal of derivational affixes.


 
Document classification via a logistic regression model:
With X and y as our feature matrices of TD-IDF values and target vector of sentiment values respectively, we are ready to split our dataset into training and test sets. Then, we will fit our training set into a Logistic Regression model.
Note that instead of manually hyperparameter tuning our model, we’re using LogisticRegressionCV to specify the number of cross-validation folds we want to do to tune the hyperparameter — that is 5-fold cross-validation.
 
 
After applying logistic regression model to the documents then the model is loaded from the disk and the accuracy is calculated using metric measures.


Model accuracy:


 89.9% accuracy is a pretty good score for relatively simple models.







Another way of looking at the accuracy of the classifier is through a confusion matrix.
 
The first row is for reviews which their actual sentiment values in the test set are 1. As you can calculate, out of 25,000 reviews, the sentiment value of 12,473 of them is 1; and out of these 12,473, the classifier correctly predicted 11,209 of them as 1.
It means, for 11,209 reviews, the actual sentiment values were 1 in the test set, and the classifier also correctly predicted those as 1. However, while the actual labels of 1,264 reviews were 1, the classifier predicted those as 0, which is pretty good.
What about the reviews with the sentiment value of 0? Let’s look at the second row. It looks like there were a total 12,527 reviews which their actual sentiment values were 0.
The classifier correctly predicted 11,193 of them as 0, and 1,334 of them wrongly as 1. So, it has done a good job in predicting the reviews with sentiment values of 0.
A good thing about confusion matrix is that it shows the model’s ability to correctly predict or separate the classes. In specific cases of a binary classifier, such as this example, we can interpret these numbers as the count of true positives, false positives, true negatives, and false negatives.




NAIVE BAYES MODEL ANALYSIS AND PERFORMANCE:
Naive Bayes it's a popular and easy to understand Supervised Probabilistic classification algorithm. The Naive Bayes Algorithm is based on the Bayes Rule which describes the probability of an event, based on prior knowledge of conditions that might be related to the event.
 
 
0.81656 accuracy is a pretty good score!
Advantages and disadvantages of Naive Bayes
Major advantages:
Simplicity
Requires small training set
Computationally fast
Scales linearly with the number of features and training examples
Disadvantages:
Strong feature independence assumption which rarely holds true in the real world. Remember, it's naive
May provide poor estimates, based on its independence assumption
