# Covid-19-Twitter-Sentimental-Analysis
![image](https://user-images.githubusercontent.com/119458164/236449588-ca2d0e13-82c2-49c8-b3ee-a66b9163697a.png)

[Medium link](https://medium.com/@kwasiasomani85/sentiment-analysis-on-covid-19-tweets-3d45bf7bb34c)

[Huggingface link](https://huggingface.co/spaces/Kwasiasomani/Streamlit-Sentimental-Analysis)


Hello there! This is the Github repository for Covid19 - The Visualisation Dashboard of COVID-19 Twitter Sentiment Analysis. This project is based on analyzing the sentiment of tweets which help in understanding the pulse of the nation towards the pandemic. 
Recently, the number of tweets on COVID-19 are increasing at an unprecedented rate by including positive, negative and neutral tweets. This diversified nature of tweets has attracted the researchers to perform sentiment analysis and analyze the varied emotions of a large public towards COVID-19. The traditional sentiment analysis techniques will only find out the polarity and classify it as either positive, negative or neutral tweets
In this project the data for training was obtained from Zindi. Transfer learning approach was adopted in building the models in the project. Two deep learning models were used,namely:

1. RoBERTa - For Sentiment Analysis
2. Distillbert - For Sentiment Triggers Extraction.

# Development Phase
For the Model Training and Validation Pytorch approach was used. The development phase of the project is divided in 5 phases:

  1. [Data Collection and Cleaning](https://github.com/kwasiasomani/Covid-19-Sentimental-Analysis/blob/main/Notebook/Sentimental_Analysis_using_DistilBERT_Model.ipynb)
2. [Roberta](https://github.com/kwasiasomani/Covid-19-Sentimental-Analysis/blob/main/Notebook/Sentimental_Analysis_using_Roberta_base_model.ipynb) & [Distilbert Model Sentiment Extractor and Sentiment Analyzer](https://github.com/kwasiasomani/Covid-19-Sentimental-Analysis/blob/main/Notebook/Sentimental_Analysis_using_DistilBERT_Model.ipynb).
3. [Development and Deployment of Streamlit App using HuggingFace](https://huggingface.co/spaces/Kwasiasomani/Streamlit-Sentimental-Analysis)
4. [Development and Deployment of Gradio App using HuggingFace](https://huggingface.co/spaces/Kwasiasomani/Gradio-Sentimental-Analysis)



# Data Creation and Data Cleaning
Data set was downloaded from Zindi website. Upon inspection, it was found that safe_text did have missing values so further inspection was taken to check if it has plenty missing values and check the rows and columns affected. So the missing values was dropped for best the train and test data.


# Roberta Model Training
Transfer learning methods were implemented to carry out sentiment analysis. Sentiment Analysis of Tweets was carried out by integrating and using  the Huggingface Transformer Library. Further Slanted Triangular Learning Rates, The trainin arguments was defined ,epochs was set to 10, load the best model and it should return RMSE scores.The least scores gives the best prediction. The Data obtained from the previous process was then tokenised and passed through the model for Sentiment analysis. This yielded a model with an RMSE of 68% over the data set

# Distilbert Model Training
Transfer learning methods were implemented to carry out sentiment analysis. Sentiment Analysis of Tweets was carried out by integrating and using  the Huggingface Transformer Library. Further Slanted Triangular Learning Rates, The trainin arguments was defined ,epochs was set to 10, load the best model and it should return RMSE scores.The least scores gives the best prediction. The Data obtained from the previous process was then tokenised and passed through the model for Sentiment analysis. This yielded a model with an RMSE of 65% over the data set


# HuggingFace
A huggingface  was used for setting up website routing. It is used to integrate the back end machine learning models with the dashboard.
We can use HuggingFace Transformers for performing easy text summarization. We'll structure things as follows. First of all, we'll be looking at how Machine Learning can be useful to summarizing text. Subsequently, we looked at how summarization can be performed with a pretrained Transformer. We'll look at Transformers, BERT and Roberta. Transformer models have proven to be exceptionally efficient over a wide range of ML tasks, including Natural Language Processing (NLP), Computer Vision, and Speech

# Gradio app

![1](https://user-images.githubusercontent.com/119458164/236497416-29d23043-768c-401a-9da2-5a8b368462fc.PNG)

# Streamlit

![1](https://user-images.githubusercontent.com/119458164/236504795-d299f4f3-2b29-4ca7-82ae-c0c60897ccb1.PNG)
![2](https://user-images.githubusercontent.com/119458164/236504862-6d6013e9-b37e-4df0-a322-01a6e9e21f69.PNG)
![4](https://user-images.githubusercontent.com/119458164/236504920-ac9c096d-0fc5-4121-a319-3d1d68400fba.PNG)
![6](https://user-images.githubusercontent.com/119458164/236505114-d58059c2-fe99-4003-8963-1f72e4fc4955.PNG)





