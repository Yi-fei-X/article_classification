# Article Classification

The goal of this assignment is to classify documents in a corpus.  

#### Dataset description:  
Raw training data (https://wikispaces.psu.edu/download/attachments/395383213/news-train.csv?api=v2) with labels:  
The dataset contains raw text of 1490 news articles and the article category. Each row is a document.  
The raw file is a .csv with three columns: ArticleId, Text, Category   
The “Category” column are the labels you will use for training

Raw test data (https://wikispaces.psu.edu/download/attachments/395383213/news-test.csv?api=v2) without labels  
This dataset contains raw text of 736 news articles. Each row is a document.  
The raw file is a .csv with two columns: ArticleId,Text.  
The labels are not provided

#### My job:

1. Preprocess the raw training data.  

2. Evaluate the decision tree model on your pre-processed data.  

3. Evaluate random forests model on pre-processed training data.  

4. Evaluate XGBoost on pre-processed training data.

5. Predict the labels for the testing data (using raw training data and raw testing data). 
