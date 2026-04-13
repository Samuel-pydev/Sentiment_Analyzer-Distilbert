from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline 

model = AutoModelForSequenceClassification.from_pretrained("./my_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./my_sentiment_model")

sentiment = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

while True:
    user_input = input("Enter Text: ")

    if user_input.lower() == 'quit':
        break

    results = sentiment(user_input)

    print(" Results: ", results)

    

