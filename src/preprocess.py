"""
preprocess.py clean and prepares the news articles for futher analysis.
"""

class Preprocessor:
    def __init__(self, texts):
        """initialise the preprocessor with a list of texts."""
        self.texts = texts

    def clean(self):
        """remove newlines and extra spaces from each text."""
        cleaned = []
        for t in self.texts:
            try:
                if isinstance(t, str): # only process strings
                    new_text = t.strip().replace("\n", " ")
                    cleaned.append(new_text)
                else:
                    # skip non-string entries but print a small notice to flag
                    print(f"Skipping non-string value: {t}")
            except Exception as e:
                print(f"Something went wrong with cleaning text --> Error message: {type(e).__name__} - {e}.")
        return cleaned

if __name__ == "__main__":
    # default example when running this file directly
    sample_texts = [
        "  Artificial Intelligence is evolving fast.\n",
        "AI models learn from data.\n\n",
        123, # invalid type
        None  # invalid type
    ]

    processor = Preprocessor(sample_texts)
    cleaned_texts = processor.clean()

    print("\n Processed texts:")
    for text in cleaned_texts:
        print("-", text)