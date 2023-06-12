from transformers import pipeline


class SentimentAnalyzer:
    """Class for analyzing the sentiment of sentences
    """

    def __init__(self) -> None:
        """initializes the class with sentiment analysis pipeline using the distilbert-base-uncased-finetuned-sst-2-english model
        """
        self.analyzer = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def score_sentiment(self, sentence: str) -> float:
        """Uses the analyzer to analyze the sentiment of the provided sentence

        Parameters
        ----------
        sentence : str
            a short sentence to be analyzed

        Returns
        -------
        float
            score of the sentiment from 0 to 1. Below 0.5 is negative, above is positive. 0.5 is neutral
        """
        return self.analyzer(sentence)[0]

    def get_sentiment(self, sentence: str) -> str:
        """returns the label of the sentiment provided

        Parameters
        ----------
        sentence : str
            a short sentence to be analyzed

        Returns
        -------
        str
            label of the sentiment wether it is positive, negative, or neutral
        """
        sentiment_score = self.score_sentiment(sentence)
        return sentiment_score['label']


if __name__ == "__main__":
    sentence = "I hate you"
    sentiment_analyzer = SentimentAnalyzer()
    print(sentiment_analyzer.get_sentiment(sentence))