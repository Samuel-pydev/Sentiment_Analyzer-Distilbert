from datasets import load_dataset
dataset = load_dataset("mteb/tweet_sentiment_multilingual", "english")
print(dataset)
print(dataset['train'][0]) 
