from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline 

model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model_3class")
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model_3class")

sentiment = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

test_sentences = [
    "this movie was absolutely amazing",
    "the film was not good at all",
    "what is that",
    "what other fetish do you have?",
    "i did not enjoy this at all",
    "it was okay i guess",
]

for sentence in test_sentences:
    result = sentiment(sentence)
    print(f"{sentence} --> {result}")



    

