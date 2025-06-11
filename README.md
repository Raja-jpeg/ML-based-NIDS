# Machine Learning-based Network Intrusion Detection System

### Detailed Breakdown of the Code and Model Descriptions:

#### 1. **Data Preprocessing:**
   - **Feature Scaling:** 
     - The first step in the code involves scaling the features of the dataset using `StandardScaler`. Scaling standardizes the dataset, transforming the data into a distribution with a mean of 0 and a standard deviation of 1. This is important for machine learning algorithms that are sensitive to the scale of the input data, such as SVM, KNN, and Decision Trees.
   - **Dimensionality Reduction (PCA):**
     - **Incremental PCA** is then applied to reduce the dimensionality of the data. PCA transforms the original feature space into a smaller set of uncorrelated features (principal components), preserving the maximum amount of variance. Incremental PCA is used here because it can handle large datasets in chunks, rather than loading the entire dataset into memory.
   - **Label Encoding for Binary Classification:**
     - For binary classification, the dataset is split into benign (`BENIGN`) and malicious (non-benign) traffic. The labels are then encoded as 0 for benign traffic and 1 for malicious traffic. This binary labeling is typical when building models that focus on detecting whether traffic is benign or malicious.

#### 2. **Balancing the Dataset:**
   - **SMOTE (Synthetic Minority Over-sampling Technique):**
     - The code then applies **SMOTE** to balance the dataset by upsampling the minority class (malicious traffic) in multi-class classification tasks. SMOTE generates synthetic samples for the minority class by interpolating between existing instances. This helps prevent the model from being biased toward the majority class (benign traffic).
   - **Resampling for Binary Classification:**
     - For the binary classification task, the code balances the dataset by randomly sampling the benign class to match the number of malicious instances. After this, a smaller subset of 15,000 rows is sampled for training. The distribution of the classes (benign and malicious) is then printed to ensure balance.

---

### 3. **Model Descriptions:**

#### **Support Vector Machine (SVM) for Binary Classification:**
   - **Overview:** 
     - **SVM** is a supervised learning algorithm primarily used for classification tasks. It works by finding the optimal hyperplane that maximizes the margin between the two classes. In this case, the model uses the RBF (Radial Basis Function) and polynomial kernels to map the data into higher-dimensional space and find the best separating hyperplane.
   - **Key Parameters:**
     - `kernel`: The choice of kernel defines the transformation of the feature space. Here, `rbf` and `poly` kernels are used.
     - `C`: Controls the trade-off between achieving a larger margin and minimizing classification error.
     - `gamma`: Controls the influence of a single training example on the decision boundary.
     - `random_state`: Ensures reproducibility of results by fixing the random seed.

   - **Model Application:** 
     - After splitting the dataset into benign and malicious traffic, the dataset is balanced, and SVM is trained on the binary classification data. Cross-validation is then performed to evaluate the performance, with the model showing a strong mean cross-validation score of 0.96, indicating good generalization.

---

#### **Decision Tree Classifier:**
   - **Overview:** 
     - **Decision Tree** is a non-linear classifier that splits the data into distinct regions based on feature values, creating a tree-like structure of decisions. Each node represents a decision rule, and the leaves represent the outcome (class).
   - **Key Parameters:**
     - `max_depth`: Limits the maximum depth of the tree to prevent overfitting. A depth of 8 is used here, which is a reasonable choice to avoid overly complex trees.
   - **Model Application:**
     - The dataset is filtered to select only attack types with more than 1950 occurrences, ensuring sufficient data for training. SMOTE is applied to upsample the minority classes, and the model is trained using the balanced dataset. Cross-validation results show a high mean score of 0.97, suggesting excellent model performance.

---

#### **Random Forest Classifier:**
   - **Overview:** 
     - **Random Forest** is an ensemble method based on decision trees. It builds multiple decision trees using random subsets of the data and features, then combines the predictions to improve accuracy and reduce overfitting.
   - **Key Parameters:**
     - `n_estimators`: Specifies the number of trees in the forest. A smaller number, like 15, is chosen here for faster training and testing.
     - `max_depth`: Limits the depth of each tree to prevent overfitting.
     - `max_features`: Specifies the number of features to consider when looking for the best split, helping to reduce overfitting.
   - **Model Application:**
     - After applying SMOTE to balance the dataset, Random Forest is trained on the balanced data, and cross-validation results show an impressive mean score of 0.98, suggesting it is a highly accurate model for this task.

---

#### **K-Nearest Neighbors (KNN) and Random Forest Ensemble Classifier:**
   - **Overview:**
     - **KNN** is a simple classifier that assigns a class based on the majority vote of its k nearest neighbors. It works well when the decision boundary is complex and non-linear.
     - **Ensemble Learning** involves combining multiple models to improve performance. Here, KNN and Random Forest are combined using a **Voting Classifier**, which makes predictions based on the majority vote of the individual classifiers.
   - **Key Parameters:**
     - `n_neighbors`: Specifies the number of nearest neighbors to consider in KNN. A value of 5 is used here.
     - `voting`: Defines the strategy for combining predictions from multiple models. The ‘soft’ voting approach uses probabilities, taking the average of the predicted probabilities from each model.
   - **Model Application:**
     - After training the individual models (KNN and Random Forest), the **Voting Classifier** is used to combine them. This ensemble model is trained on the data and achieves an impressive accuracy of 0.984, with a classification report showing high precision, recall, and f1-scores for all attack types.

---

### 4. **Evaluation and Performance:**
   - **Cross-Validation Results:**
     - All models (SVM, Decision Tree, Random Forest, and the KNN + Random Forest Ensemble) are evaluated using **cross-validation** with 5 folds. This ensures the models' robustness by splitting the dataset into 5 different training and validation sets.
   - **Classification Report:**
     - The **Voting Classifier** achieves excellent classification performance, with high **precision**, **recall**, and **f1-scores** for each attack type. The accuracy score of 0.98 shows that the ensemble model is highly reliable in identifying both benign and malicious traffic.
   
---

### 5. **Conclusion:**
   - The implementation uses multiple machine learning models to tackle the **multi-class classification** problem of network intrusion detection. By leveraging techniques like **SMOTE** for balancing the dataset and **Incremental PCA** for dimensionality reduction, the models are trained on well-prepared data, ensuring they capture the complex relationships between features and attack types.
   - Among the models, the **Random Forest** and **KNN + Random Forest Ensemble** perform exceptionally well, with the ensemble method achieving the highest accuracy. However, all models show strong performance, with **SVM** and **Decision Tree** also delivering impressive results. This approach provides a comprehensive strategy for building an effective intrusion detection system.
