# E-Commerce Reviews Sentiment Analysis using Pairwise Ranking

## Business Context

In the competitive landscape of e-commerce, customer reviews play a pivotal role in shaping purchase decisions. These reviews serve as critical resources for potential buyers to assess product quality and suitability. However, the overwhelming volume of reviews often complicates the decision-making process, making it challenging for customers to identify the most valuable feedback.

---

## Objective

The objective of this project is to develop a model that employs **pairwise ranking** to prioritize product reviews, ensuring that the most relevant reviews are emphasized while less pertinent or irrelevant reviews are downgraded.

---

## Tech Stack

- **Programming Language**: [Python](https://www.python.org/)
- **Libraries and Tools**:
  - [`pandas`](https://pandas.pydata.org/) for data manipulation.
  - [`scikit-learn`](https://scikit-learn.org/) for feature extraction and classification.
  - [`nltk`](https://www.nltk.org/) and [`spaCy`](https://spacy.io/) for natural language processing.
  - [`TextBlob`](https://textblob.readthedocs.io/en/dev/) for sentiment analysis and language detection.
  - [`profanity-check`](https://github.com/vzhou842/profanity-check) for identifying inappropriate language.

---

## Approach

### 1. Data Preprocessing
- **Language Detection**:
  - Identify the language of each review using `TextBlob` or `spaCy`.
  - Filter out non-target language reviews.

- **Gibberish Detection**:
  - Implement methods to detect and exclude incoherent or nonsensical content.

- **Profanity Detection**:
  - Use `profanity-check` to flag and handle reviews with inappropriate language.

### 2. Feature Extraction
- Extract meaningful features from reviews, such as:
  - Sentiment polarity and subjectivity scores.
  - Word and sentence count.
  - Presence of specific keywords or product features.

### 3. Pairwise Review Ranking
- Compare reviews in pairs to evaluate their relevance.
- Use a ranking algorithm to prioritize reviews that:
  - Provide detailed and useful insights.
  - Are coherent and relevant to the product.

### 4. Classification
- **Relevance Classification**:
  - Train a classifier to categorize reviews as relevant or irrelevant.

- **Generate Ranked List**:
  - Create a ranked list of reviews for each product, positioning the most relevant reviews at the top.

---

## Project Structure

```plaintext
.
├── data/                                 # Contains raw and preprocessed data files.
├── src/                                  # Source code folder.
│   ├── data_preprocessing.py             # Scripts for preprocessing tasks (language, gibberish, profanity detection).
│   ├── feature_extraction.py             # Functions for extracting review features.
│   ├── ranking_algorithm.py              # Implementation of pairwise ranking.
│   ├── classification.py                 # Scripts for training and evaluating classifiers.
│   ├── engine.py                         # Main script to execute the pipeline.
├── output/                               # Stores processed results and ranked review lists.
├── requirements.txt                      # File listing dependencies and versions.
└── README.md                             # Project documentation.
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. Install Dependencies

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

### 3. Run the Project

Execute the pipeline by running the `engine.py` script:

```bash
python src/engine.py
```

### 4. Explore Results

- Check the `output/` folder for ranked review lists and processed data.
- Analyze classifier performance and feature extraction insights.

---

## Results

- **Improved Review Relevance**:
  - Generated ranked lists of reviews that prioritize helpful and relevant feedback.
- **Enhanced User Experience**:
  - Simplified the decision-making process for potential buyers.
- **Robust Preprocessing Pipeline**:
  - Effectively filtered irrelevant, gibberish, and profane content.

---

## Feature of this project:

- **User-Centric**: Designed to improve the shopping experience by prioritizing meaningful reviews.
- **Advanced NLP Techniques**: Leverages natural language processing for sentiment analysis and ranking.
- **Scalable**: Applicable to various e-commerce platforms with minimal adjustments.

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:

```bash
git checkout -b feature-name
```

3. Commit your changes:

```bash
git commit -m "Add feature"
```

4. Push your branch:

```bash
git push origin feature-name
```

5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For any questions or suggestions, please reach out to:

- **Name**: Abhinav Navneet
- **Email**: mailme.AbhinavN@gmail.com
- **GitHub**: [AjNavneet](https://github.com/AjNavneet)

---

## Acknowledgments

Special thanks to:

- [TextBlob](https://textblob.readthedocs.io/en/dev/) for sentiment analysis and language detection.
- [profanity-check](https://github.com/vzhou842/profanity-check) for identifying inappropriate language.
- [NLTK](https://www.nltk.org/) and [spaCy](https://spacy.io/) for natural language processing.
- The Python open-source community for exceptional tools and libraries.

---
