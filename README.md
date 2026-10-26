
# Spam vs Non-Spam Message Classifier

## Objective
Build a machine learning system to classify SMS messages
as spam or non-spam.

## Technologies
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Chi-square feature selection
- Multinomial Naive Bayes

## Project workflow
1. Load labeled SMS messages.
2. Convert messages into TF-IDF features.
3. Select important features using Chi-square.
4. Train a Naive Bayes classifier.
5. Evaluate the classifier using test data.
6. Save the trained model.
7. Predict new messages.

## Installation
Install the required packages:

python -m pip install -r requirements.txt

## Prepare the dataset
Place the original UCI dataset file named
SMSSpamCollection in the project folder.

Convert it to CSV:

python convert_dataset.py

## Train the model
python train.py

## Predict messages
python predict.py

## Expected output
The program predicts either SPAM or NON-SPAM.
Training also displays accuracy, precision, recall,
F1-score, and a confusion matrix.

## Limitations
Performance depends on the training dataset.
The small demonstration dataset must not be used
to claim reliable model accuracy.
