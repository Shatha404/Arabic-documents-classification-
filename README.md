# Document Classification for Arabic Documents

This project focuses on classifying Arabic documents into predefined categories (Economy, International News, Local News, Sports) using machine learning techniques. It includes steps such as data preprocessing, feature engineering, model training, evaluation, and deployment through a web application.

## Dataset

- **Dataset Name**: Khaleej-2004
- **Statistics**:
  - **Number of Categories**: 4
  - **Number of Documents**: 5690
  - **Categories**: Economy, International News, Local News, Sports

## Data Preprocessing

Steps performed:
1. **Text Cleaning**:
   - Removed punctuation, numbers, and extra spaces
   - Normalized Arabic letters (e.g., `ا` for `إ`, `أ`, `آ`)
2. **Tokenization**:
   - Split text into tokens using `.split()`
3. **Vectorization**:
   - Used TF-IDF (Term Frequency-Inverse Document Frequency)
4. **Handcrafted Features**:
   - Added features: word count, character count, and keyword presence/frequency
5. Combined sparse matrices using `scipy.sparse` for memory efficiency.

## Feature Selection

- Used `SelectKBest` with the chi-squared test to select the top 20 features.

## Model Selection and Training

### Classifiers Used:
1. **Naive Bayes**
2. **LGBM Classifier**
3. **XGBoost Classifier**

### Evaluation Metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Results (Summary):
#### Naive Bayes
- **Accuracy**: 91%
- Best performance for International News and Sports categories.

#### LGBM Classifier
- **Accuracy**: 94%
- Strong performance across all categories but some misclassifications in Economy.

#### XGBoost Classifier
- **Accuracy**: 94%
- Balanced performance but slightly lower recall for Economy.

## Prediction

![image](https://github.com/user-attachments/assets/17261ce6-5f43-4124-985b-745943e6ec24)

## Deployment

- A Flask-based web application was developed.
- **API Endpoint**: `/predict`
- **GUI**: Interactive form for text input and category prediction.

![Screenshot 2024-12-29 212520](https://github.com/user-attachments/assets/24e02328-00fa-44cf-b9aa-3968337b7f8b)



## Conclusion

While LGBM and XGBoost achieved higher overall accuracy, Naive Bayes was chosen for its simplicity and reliability, particularly in Economy and International News categories.

