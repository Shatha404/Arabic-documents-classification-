# Arabic-documents-classification-
 classify Arabic documents into predefined categories.

The objective of this project is to classify Arabic documents into predefined categories (Economy, International news ,Local News ,Sports) using machine learning. The project involves preprocessing Arabic text, extracting features, training multiple classifiers, evaluating their performance, and deploying the model via a web application.
________________________________________
### Dataset
- Dataset Name: Khaleej-2004
- Statistics: 
  - Number of Categories: 4.
  - Number of Documents: 5690.
  - Categories: Economy, International News,Local News,Sports.
 
### Data Preprocessing
•	Steps Performed: 
1.	Text Cleaning using re (Regular Expressions): 
	Removed punctuation.
	Removed numbers.
	Removed the extra spaces.
	Normalized Arabic letters (e.g., ا for إ, أ, آ).
2.	Tokenization: 
	Split text into tokens (words) using the method .split().

3.	Vectorization: 
	Used TF-IDF (Term Frequency-Inverse Document Frequency) to represent text numerically.
4.	Handcrafted Features: 
	Added three features: word count, character count, keyword presence, and frequency in the text.
5.	Combines sparse matrices (TF-IDF features and handcrafted features) using scipy.sparse for memory-efficient processing.
________________________________________
4. Feature Engineering
•	SelectKBest to select the top k=20 features and chi-squared for scoring function to Measure the chi-squared statistic between features and target labels to evaluate feature importance.
•	Applied feature selection using tool SelectKBest with the chi-squared test from scikit-learn to select the top 20 features.
•	Keywords used for handcrafted features include terms like “الاقتصاد” (economy), “الرياضة” (sports), etc.
________________________________________

5. Model Selection and Training
 Classifiers Used Based on their accuracy after applying lazy classifier the top three accuracy was:
1.	Naive Bayes
2.	LGBM Classifier
3.	XGBoost Classifier
•	Performance Metrics: 
o	Accuracy
o	Precision
o	Recall
o	F1-score
o	Classification Report


