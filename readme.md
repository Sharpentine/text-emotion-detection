# Emotion Detection with SVM

This project demonstrates emotion detection from text using a Support Vector Machine (SVM) classifier. The model is trained on the 'emotions' dataset and can predict six basic emotions: sadness, joy, love, anger, fear, and surprise.

## Dataset

The project uses the 'emotions' dataset, which contains text samples labeled with one of the six emotions. 

## Methodology

1. **Data Loading and Exploration:** The dataset is loaded and explored using Pandas and Matplotlib to understand the distribution of emotions and text lengths.
2. **Data Preprocessing:**
    - The text data is converted into numerical features using TF-IDF vectorization.
    - The numerical features are scaled using StandardScaler.
3. **Model Training:** A LinearSVC model is trained on the preprocessed training data.
4. **Model Evaluation:** The model's performance is evaluated using accuracy score and classification report on the test data.
5. **Prediction:** The trained model is used to predict the emotion of a new text input.

## Requirements

- Python 3.x
- Pandas
- Matplotlib
- Scikit-learn

## Usage

1. Install the required libraries.
2. Load the 'emotions' dataset.
3. Run the provided code to train and evaluate the SVM model.
4. Use the trained model to predict emotions from new text input.

## Results

The model achieves an accuracy of 0.8984429356301432 on the test data. The classification report provides detailed information about the model's performance for each emotion.

## Label Encoding

The following labels are used to represent emotions:

- 0: sadness
- 1: joy
- 2: love
- 3: anger
- 4: fear
- 5: surprise

## Contributing

Contributions to this project are welcome. Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License.
