
# Natural_Text_Processing_movie_review

This repository contains a project completed for the **Natural Language Processing** course. The project focuses on applying various Natural Language Processing (NLP) techniques and building a **Naïve Bayes classifier** to predict the sentiment (positive or negative) of movie reviews.

The code was developed using **Google Colab**, utilizing its runtime environment for execution.

---

## Dataset Information

- **Location:** My Drive/Colab Notebooks/NLP
- **Filename:** `moviereviews.csv`
- **Columns:**
  - `label`: Sentiment of the review (`pos` for positive, `neg` for negative).
  - `review`: Text of the movie review written in natural human language.

---

## Project Requirements

The main objective was to apply the **Multinomial Naïve Bayes** algorithm to classify the sentiment of the reviews and evaluate the model using different metrics.

The following NLP preprocessing techniques were applied as per the provided project requirements:

- **Tokenization**: Breaking down text into individual words or tokens.
- **Case Folding**: Converting all words to lowercase to maintain consistency.
- **Synonym Substitution**: Replacing words with their synonyms using WordNet.
- **Stemming**: Reducing words to their root forms using PorterStemmer.
- **Lemmatization**: Converting words to their base or dictionary form using spaCy.
- **Punctuation Removal**: Removing all punctuation symbols from the text.
- **Stop Words Removal**: Removing common, less meaningful words (like "the", "and", etc.).
- **Vector Semantics**: Converting text into numeric form using TF-IDF Vectorizer.

Each preprocessing step was completed in **separate cells** as per the assignment instructions.

---

## Implementation Details

- **Programming Language**: Python
- **Development Environment**: Google Colab
- **Libraries Used**:
  - `pandas`
  - `numpy`
  - `nltk`
  - `spacy`
  - `scikit-learn`
  - `matplotlib`
- **Machine Learning Model**: 
  - **Multinomial Naïve Bayes** from `scikit-learn`

---

## Model Evaluation

After training the model, the following evaluation metrics were calculated:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix Visualization**

---

## Important Notes

- Only libraries discussed in the course were used.
- No extra or unauthorized libraries were installed or utilized.
- Entire coding was completed and executed successfully in Google Colab.

---

## Repository Structure

Natural_Text_Processing_movie_review/ │ ├── moviereviews.csv          # Dataset ├── Natural_Text_Processing_movie_review.ipynb   # Colab Notebook containing the code └── README.md                 # This file

---

## Conclusion

This project demonstrates the complete workflow of applying Natural Language Processing techniques to a real-world dataset and building a basic yet effective machine learning model for sentiment analysis using Naïve Bayes classification.

---

## Contributors

- [**Basharul Alam Mazu** ](https://github.com/basharulalammazu)
- [**Tanjim Rahman**](https://github.com/mdtanjimrahman)
