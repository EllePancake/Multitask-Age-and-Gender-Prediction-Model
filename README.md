# **Multitask Age and Gender Prediction Model**

This repository contains the implementation of a multitask deep learning model that predicts both **age** and **gender** from facial images. The model is based on a ResNet-18 architecture, and multiple versions were trained to explore different improvements and techniques to enhance performance and mitigate biases.

## **Table of Contents**
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Data Description](#data-description)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Model Versions](#model-versions)
6. [Performance Evaluation](#performance-evaluation)
7. [Bias and Ethical Concerns](#bias-and-ethical-concerns)
8. [Installation and Usage](#installation-and-usage)
9. [Future Work](#future-work)

## **Introduction**

This project aims to develop a **multitask deep learning model** that can accurately predict **age** and **gender** from facial images. The model uses a modified version of ResNet-18 to perform both tasks simultaneously. Multiple versions of the model were created with different preprocessing techniques, learning rates, and class rebalancing methods to understand their impact on performance and biases.

## **Project Structure**

- `data/`: Contains the train, validation, and test datasets.
- `models/`: Directory for storing model checkpoints and saved versions.
- `notebooks/`: Jupyter notebooks used for Exploratory Data Analysis (EDA) and result visualization.
- `src/`: Python scripts for model training, evaluation, and data loading.
- `README.md`: Project documentation.

## **Data Description**

The dataset consists of images of human faces annotated with **age** and **gender** labels. The dataset is divided into three subsets:
- **Training Set**: Used to train the model.
- **Validation Set**: Used to evaluate the model's performance during training.
- **Test Set**: Used to evaluate the final performance of the trained models.

## [**Exploratory Data Analysis (EDA)**]()

Exploratory Data Analysis was performed to better understand the dataset's distribution and identify potential issues related to class imbalance or biases. Key insights from the EDA include:

- **Class Distribution**:
  - **Gender**: The dataset is slightly imbalanced with more male images compared to female images.
  - **Age**: There is a significant imbalance across age groups, with a large number of samples representing younger ages (0-30) and a much smaller representation of elderly individuals (70+).
  
- **Visual Analysis**:
  - **Gender Analysis**: The dataset contains images from diverse backgrounds, but some gender-stereotypical features (e.g., hair length, clothing) could affect the model's decision-making process.
  - **Age Analysis**: The dataset includes images with varying lighting conditions and facial expressions, which may affect age prediction accuracy.
  
- **Correlation Analysis**:
  - Correlation between age and image features showed that facial textures (wrinkles, skin tone) are relevant for age prediction, while the model may rely on non-facial features like hair for gender prediction, which could lead to bias.

## **Model Versions**

### [**Base Model**](https://github.com/EllePancake/Multitask-Age-and-Gender-Prediction-Model/blob/main/modeling/models/base_model.py)
- A multitask ResNet-18 model was used, with separate heads for age and gender prediction.
- Loss functions:
  - **Age**: Mean Squared Error (MSE)
  - **Gender**: Cross Entropy Loss

### **Model Variants**
- [**Version 1 (v1)**](): Base model with minimal preprocessing (only conversion to tensor).[Training completed here.](https://github.com/EllePancake/Multitask-Age-and-Gender-Prediction-Model/blob/main/modeling/experiments/train_v1.ipynb)
- [**Version 2 (v2)**](): Added normalization for input images.[Training completed here.](https://github.com/EllePancake/Multitask-Age-and-Gender-Prediction-Model/blob/main/modeling/experiments/train_v2.ipynb)
- [**Version 3 (v3)**](): Reduced the learning rate to `0.0001` to stabilize training.[Training completed here.](https://github.com/EllePancake/Multitask-Age-and-Gender-Prediction-Model/blob/main/modeling/experiments/train_v3.ipynb)
- [**Version 4 (v4)**](): Implemented a **WeightedRandomSampler** to address class imbalance in the training set.[Training completed here.](https://github.com/EllePancake/Multitask-Age-and-Gender-Prediction-Model/blob/main/modeling/experiments/train_v4.ipynb)

## [**Performance Evaluation**](https://github.com/EllePancake/Multitask-Age-and-Gender-Prediction-Model/blob/main/results%20and%20conclusion.ipynb)

### **Gender Prediction**:
- **Best Version**: **v2** achieved the highest gender accuracy (0.904) on the test set.
- **Worst Version**: **v3** had the lowest accuracy due to potential underfitting with the reduced learning rate.

### **Age Prediction**:
- **Evaluation Metric**: Mean Absolute Error (MAE)
- **Best Version**: **v2** also performed best on age prediction, with a MAE of **5.1 years**.
- **Age Group Performance**: Younger age groups (0-30) were predicted more accurately, while elderly individuals had higher errors.

### **Overall Performance**:
- **Balanced Trade-off**: Version 2 provided the best balance between age and gender prediction performance.

## **Bias and Ethical Concerns**

### **Identifying Biases**
- **Age Bias**: The model performed poorly for elderly individuals (age 70+), indicating an age bias due to underrepresentation.
- **Gender Bias**: Gender prediction relied on stereotypical features, such as hair length, which led to biases in classification for individuals who do not conform to traditional gender presentations.

### **Common Bias Sources**
- **Dataset Imbalance**: The dataset had a significant imbalance across both age and gender, leading to biased performance.
- **Training and Features**: The model sometimes used irrelevant features (e.g., background, hair) for classification, which can reinforce existing stereotypes.

### **Mitigation Strategies**
- **Data Rebalancing**: Implementing class rebalancing using `WeightedRandomSampler` (v4) partially addressed bias but did not fully solve it.
- **Future Improvements**: Gathering more diverse training samples and implementing fairness-aware algorithms could further reduce biases.

### **Use-case Suitability**
- **Suitable Use-Cases**: The model can be used for applications such as age-based content recommendations or non-critical personalization systems.
- **Unsuitable Use-Cases**: The model should not be used in high-stakes contexts, such as hiring, healthcare, or law enforcement, due to its inherent biases.

## **Future Work**

1. **Improving Data Diversity**: Increase the dataset's diversity by adding more samples across all demographic groups, particularly older individuals and underrepresented gender expressions.
2. **Advanced Bias Mitigation Techniques**: Implementing adversarial debiasing and regularizing model focus to rely on unbiased features.
3. **Post-training Audits**: Conduct fairness audits regularly to ensure responsible and ethical deployment.
4. **Context-Specific Model Adaptation**: Tailor the model for low-risk applications with less severe consequences of misclassification.
