from datasets import load_dataset
DATASETS='"cardiffnlp/twitter_sentiment_multilingual", "english"'

dataset = load_dataset("Sp1786/multiclass-sentiment-analysis-dataset")
print(dataset)
print(dataset['train'][0]) 
