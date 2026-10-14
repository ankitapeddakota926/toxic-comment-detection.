# 🛡️ Toxic Comment Detection System

An NLP-based Machine Learning application that identifies potentially toxic or abusive online comments using TF-IDF feature extraction and One-vs-Rest Logistic Regression. The system predicts six categories of toxicity and provides an interactive web interface built using Streamlit.

## 📌 Project Overview

Online platforms receive large numbers of comments every day. Some comments may contain insults, threats, obscene language, or other forms of harmful content. Manually reviewing every comment can be time-consuming.

The **Toxic Comment Detection System** uses Natural Language Processing (NLP) and Machine Learning to analyze text and identify potentially harmful content. It can help moderators prioritize comments for human review.

The application allows users to enter a comment and view its predicted toxicity labels through a simple web interface.

## 🎯 Objectives

- Understand NLP-based text classification for content moderation.
- Analyze and preprocess textual data.
- Examine label frequencies and class imbalance.
- Convert text into numerical features using TF-IDF.
- Implement multilabel classification using the One-vs-Rest strategy.
- Train Logistic Regression classifiers for six toxicity categories.
- Evaluate the model using precision, recall, F1-score, and multilabel metrics.
- Analyze false positives and false negatives.
- Save the trained model and preprocessing objects.
- Develop and deploy an interactive Streamlit application.

## 📊 Dataset

**Dataset:** Jigsaw Toxic Comment Classification Challenge

The project uses the Jigsaw dataset, which contains online comments and their associated toxicity labels.

Dataset columns include:

- `id` — Unique comment identifier.
- `comment_text` — Text of the comment.
- `toxic` — General toxic content.
- `severe_toxic` — Severely toxic content.
- `obscene` — Obscene language.
- `threat` — Threatening language.
- `insult` — Insulting language.
- `identity_hate` — Identity-based hateful content.

The training dataset used in this project contains approximately **159,571 comments**.

Dataset source: [Jigsaw Toxic Comment Classification Challenge — Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## 🧰 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Model development and training |
| Pandas | Dataset loading and analysis |
| NumPy | Numerical operations |
| Regular Expressions | Text preprocessing |
| Scikit-learn | Feature extraction, model training, and evaluation |
| TF-IDF | Converting text into numerical features |
| Logistic Regression | Classification algorithm |
| One-vs-Rest | Multilabel classification strategy |
| Joblib | Saving and loading trained objects |
| Streamlit | Interactive web application |
| GitHub | Source code and project version control |
| Streamlit Community Cloud | Application deployment |

## ⚙️ Project Workflow

### 1. Dataset Loading

The dataset is uploaded and extracted in Google Colab. The training CSV is loaded using Pandas, and its rows, columns, missing values, and sample comments are inspected.

### 2. Exploratory Data Analysis

The frequency of each toxicity label is calculated and visualized using a bar chart.

The dataset has class imbalance because some toxicity categories have substantially fewer positive examples than others.

### 3. Text Preprocessing

The comment text is cleaned using the following operations:

- Convert text to lowercase.
- Decode HTML entities.
- Replace URLs with a placeholder.
- Remove HTML tags.
- Normalize extra whitespace.

The cleaned comments are used for model training and prediction.

### 4. TF-IDF Feature Extraction

Term Frequency–Inverse Document Frequency (TF-IDF) converts textual comments into numerical feature vectors.

The implementation uses:

- A maximum of 50,000 features.
- Unigrams and bigrams.
- Minimum document frequency of 2.
- Sublinear term-frequency scaling.

The TF-IDF vectorizer is fitted only on the training data to prevent data leakage.

### 5. Multilabel Classification

The project uses **One-vs-Rest Logistic Regression**.

Six binary classifiers are trained, one for each toxicity category:

1. Toxic
2. Severe toxic
3. Obscene
4. Threat
5. Insult
6. Identity hate

Class weighting is used to help account for class imbalance.

A comment may receive multiple labels at the same time. For example, a comment may be classified as both toxic and insulting.

### 6. Model Evaluation

The model is evaluated using a held-out test set.

The evaluation metrics include:

- **Precision:** Measures how many positive predictions are correct.
- **Recall:** Measures how many actual positive examples are identified.
- **F1-score:** Combines precision and recall.
- **Micro F1-score:** Evaluates performance across all label decisions collectively.
- **Macro F1-score:** Gives equal importance to each toxicity category.
- **Hamming Loss:** Measures the proportion of incorrect individual label decisions.
- **Exact Match Accuracy:** Measures how many comments have all six labels predicted correctly.

False positives and false negatives are also examined to understand incorrect flags and missed harmful comments.

Actual performance values depend on the trained model and should be taken from the notebook's evaluation output.

### 7. Model Saving

The trained model and preprocessing objects are saved using Joblib.

The saved files are:

- `toxic_model.joblib`
- `tfidf_vectorizer.joblib`
- `labels.joblib`

These files allow the Streamlit application to load the trained model without retraining it.

### 8. Streamlit Application

The application provides a simple interface where users can:

- Enter or paste a comment.
- Analyze the comment.
- View the predicted toxicity categories.
- See whether any toxicity labels were flagged.
- Expand a section to view all six label predictions.

The application uses the same preprocessing steps and TF-IDF vectorizer as the training notebook.

## 🖥️ Application Output

The application provides two main types of results.

**Potentially toxic content detected**

The model predicts one or more toxicity categories and displays the flagged labels.

**No toxicity labels predicted**

The model does not flag the comment for any of the six categories.

These are model predictions, not definitive judgments. A comment that is not flagged may still contain harmful content.

## 📁 Repository Structure

The repository is organized as follows:

```text
toxic-comment-detection/
│
├── app.py
├── requirements.txt
├── toxic_model.joblib
├── tfidf_vectorizer.joblib
├── labels.joblib
└── README.md
```

The Google Colab notebook contains the dataset preparation, exploratory analysis, feature extraction, model training, and evaluation steps.

The trained model files are included in the deployment repository so the Streamlit application can load them directly.

## 🚀 Run the Application Locally

### Prerequisites

- Python 3.12
- pip
- The project files listed above

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/toxic-comment-detection.git
cd toxic-comment-detection
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

`http://localhost:8501`

## ☁️ Deployment Using Streamlit Community Cloud

The application can be deployed using Streamlit Community Cloud.

1. Upload the application and trained model files to a GitHub repository.
2. Ensure that `app.py` and `requirements.txt` are in the repository's main directory.
3. Open [Streamlit Community Cloud](https://share.streamlit.io/).
4. Sign in with GitHub.
5. Create a new app and select the repository.
6. Select the `main` branch.
7. Set the main file path to `app.py`.
8. Select a compatible Python version in the deployment settings.
9. Deploy the application and wait for the build to complete.

After successful deployment, Streamlit provides a public URL for accessing the application.

## 🧪 Testing

The application can be tested with different types of comments.

**Example 1: Friendly comment**

```text
Thank you for your help and support. You are a wonderful person, and I appreciate your kindness.
```

Expected behavior: Ideally, no toxicity labels are predicted.

**Example 2: Insulting comment**

```text
You are an idiot and nobody likes you.
```

Expected behavior: The model may predict the toxic and insult categories.

**Example 3: Neutral disagreement**

```text
I disagree with your opinion about this topic.
```

Expected behavior: Ideally, no toxicity labels are predicted.

The actual outputs may vary depending on the trained model. Testing multiple examples helps assess the application's behavior.

## ⚠️ Limitations

- The model may misunderstand sarcasm, context, slang, or indirect expressions.
- Class imbalance may affect performance on less frequent toxicity categories.
- False positives can incorrectly flag acceptable comments.
- False negatives can allow harmful comments to go undetected.
- The model's predictions should not be treated as definitive judgments.
- Human review is recommended before taking moderation action.

## 🔮 Future Enhancements

- Experiment with advanced language models such as BERT or DistilBERT.
- Tune classification thresholds for different toxicity categories.
- Improve performance on underrepresented classes.
- Add confidence scores and per-label prediction probabilities.
- Support batch comment analysis and CSV uploads.
- Add moderation history and downloadable reports.
- Improve the user interface for mobile devices.
- Evaluate the model on newer and more diverse datasets.

## 🏁 Conclusion

The Toxic Comment Detection System demonstrates how Natural Language Processing and Machine Learning can be used to support online content moderation. It combines text preprocessing, TF-IDF feature extraction, One-vs-Rest Logistic Regression, multilabel evaluation, and a Streamlit interface.

The application provides predicted toxicity categories for user-entered comments and can help prioritize potentially harmful content for human review. Its effectiveness depends on model performance, data quality, and the context of each comment.

## 👩‍💻 Project Information

**Project Name:** Toxic Comment Detection System

**Domain:** Natural Language Processing and Machine Learning

**Programming Language:** Python

**Development Environment:** Google Colab

**Application Framework:** Streamlit

**Deployment Platform:** Streamlit Community Cloud

**Dataset:** Jigsaw Toxic Comment Classification Challenge

## 📜 Disclaimer

This project is an educational machine learning prototype. It is not a perfect detector of harmful content and should not be used as the sole basis for moderation, punishment, or other consequential decisions.

## 🙏 Acknowledgements

Thanks to the creators of the Jigsaw Toxic Comment Classification dataset and the open-source communities behind Python, Pandas, NumPy, Scikit-learn, Joblib, and Streamlit.
