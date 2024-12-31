# Phishing URL Detection Using Random Forest and LLMs

## Introduction
This project tackles the important problem of detecting phishing URLs to enhance internet security. Phishing attacks exploit fraudulent websites to steal sensitive information from unsuspecting users. We employ machine learning and modern language models to classify URLs as phishing or legitimate, providing a robust and interpretable solution to this pervasive issue.

## Problem Formulation
Phishing URL detection is framed as a binary classification problem where the objective is to categorize a URL as either "Phishing" or "Legitimate."

### Objectives
1. **Accuracy**: Develop a highly accurate model to classify phishing URLs.
2. **Robustness**: Validate the model's reliability using alternative approaches.
3. **Interpretability**: Compare results from traditional machine learning (Random Forest) with those from modern Language Learning Models (LLMs).

## Dataset
We used the [Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) from the UCI Machine Learning Repository. The dataset includes features extracted from URLs, such as length, number of special characters, HTTPS usage, and domain similarity indices.

## Approach
### 1. Random Forest Classifier
The Random Forest algorithm was used as the primary machine learning model.
- **Key Characteristics**:
  - **Bagging Technique**: Combines multiple decision trees to reduce variance and improve generalization.
  - **Feature Importance**: Measures the contribution of individual features to classification accuracy.
- **Hyperparameters**:
  - Number of Trees: 200
  - Max Depth: 20
  - Minimum Samples Split: 10
  - Minimum Samples per Leaf: 4

### 2. Integration with LLMs
Due to the suspiciously perfect performance of the Random Forest model (100% accuracy), we introduced a Large Language Model (LLM) for cross-validation.

#### Usage of LLMs
We utilized Google’s Gemini LLM to analyze phishing URLs based on extracted features.
- **Prompt Engineering**: We created a structured template to guide the LLM’s evaluation.
- **Features Provided to LLM**:
  - URL characteristics (length, domain, TLD, special characters).
  - Webpage content features (presence of forms, hidden fields, favicon).
- **LLM Outputs**:
  - Assessment (SAFE or PHISHING).
  - Confidence (HIGH, MEDIUM, LOW).
  - Risk Score (0-100).
  - Reasons for classification.

#### Comparison with Random Forest
We compared the LLM’s predictions to the Random Forest results. The focus was on:
- Consistency between the two methods.
- Interpretability of LLM’s reasoning.

## Evaluation and Interpretation
### Model Performance Metrics
#### Random Forest
- **Accuracy**: 100%
- **Precision**: 99.99%
- **Recall**: 100%
- **F1-Score**: 99.99%
- **ROC-AUC**: 100%

#### Observations
The perfect accuracy raised concerns about overfitting or data leakage.

### LLM Evaluation
- **Consistency**: LLM classifications aligned with Random Forest in ~98% of cases.
- **Risk Assessment**: LLM provided nuanced insights, such as identifying borderline phishing cases based on subtle features.
- **Transparency**: The LLM justified its decisions, making it easier to interpret false positives and false negatives.

### Comparative Analysis
| Criterion                | Random Forest         | LLM                      |
|--------------------------|-----------------------|--------------------------|
| Accuracy                 | 100%                 | ~98%                    |
| Interpretability         | Feature Importance   | Detailed Reasoning      |
| Robustness               | Limited by dataset   | Explored subtle patterns |

### Key Insights
- The Random Forest model’s high accuracy might be due to dataset-specific patterns.
- The LLM introduced interpretability, highlighting potential improvements for future iterations.

## Implementation Details
### Random Forest
1. Data preprocessing: Handled missing values, scaled numerical features, and encoded categorical variables.
2. Feature selection: Reduced correlated features to improve model efficiency.
3. Model training: Used stratified train-test splits to preserve class balance.
4. Visualization: Plotted feature importance, learning curves, and confusion matrices.

### LLM Integration
1. Extracted features from URLs to create a structured JSON input.
2. Sent feature data to the LLM via API requests.
3. Parsed LLM responses to retrieve assessments, confidence levels, and risk scores.
4. Combined results with Random Forest predictions for comprehensive analysis.

## Challenges and Future Directions
### Challenges
1. **Data Bias**: High dataset specificity might inflate Random Forest performance.
2. **API Constraints**: LLM integration introduced latency and dependency on external services.

### Future Work
1. Expand the dataset to include more diverse phishing URLs.
2. Explore ensemble methods that combine Random Forest and LLM outputs.
3. Investigate alternative ML models such as XGBoost and deep learning approaches.

## Conclusion
This project successfully demonstrated the potential of combining traditional machine learning with LLMs for phishing URL detection. While the Random Forest model provided high accuracy, the LLM added interpretability and robustness, paving the way for more reliable and explainable phishing detection systems.

## References
- [UCI Phishing Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- Google Gemini API Documentation

