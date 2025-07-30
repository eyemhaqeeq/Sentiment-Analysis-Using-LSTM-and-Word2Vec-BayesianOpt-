- Sentiment Analysis Using LSTM and Word2Vec

This project performs binary sentiment classification on Amazon product reviews using a deep learning model built with LSTM (Long Short-Term Memory) networks and pre-trained Word2Vec word embeddings. The goal is to classify customer reviews as either positive or negative based on their textual content.

- Overview
The pipeline includes comprehensive data preprocessing steps such as stopword removal, tokenization, and lemmatization to prepare the data for training. Word embeddings are generated using Word2Vec, and the sequential model architecture is based on LSTM layers. The model is evaluated using key classification metrics including accuracy, precision, recall, F1-score, and confusion matrix.

- Features
Text cleaning: stopword removal, tokenization, and lemmatization

Word2Vec embedding layer for semantic representation of text

LSTM network for capturing contextual relationships in sequences

Binary classification output (positive or negative sentiment)

Evaluation metrics: accuracy, precision, recall, F1-score, confusion matrix

Graphical representation of training history and performance

- Dataset
The dataset used is a subset of the Amazon Fine Food Reviews dataset, filtered and processed to include only reviews with scores that are either clearly positive or negative.

Labeling scheme:

Reviews with scores > 3 are labeled as positive (1)

Reviews with scores < 3 are labeled as negative (0)

Reviews with a score of 3 are excluded to ensure binary classification

- Implementation
Text is first cleaned and lemmatized, then tokenized into sequences. These sequences are padded to ensure uniform input length for the model. Word embeddings are trained using the Gensim Word2Vec model. The LSTM model is constructed using TensorFlow/Keras and compiled with binary crossentropy loss and the Adam optimizer.

- Model Training and Testing
Training and testing were conducted on a sample of cleaned reviews. The model was trained for a specified number of epochs using an 80/20 train-test split. Performance metrics were evaluated on the test set.

- Results(for sample 15000 reviews)
Accuracy: 0.8976666666666666
Training Time: 168.220472574234
Testing Time: 10.29760217666626

Classification metrics showed strong performance across both classes, with balanced precision and recall. Visualizations including accuracy and loss curves, as well as a confusion matrix heatmap, were generated to analyze model performance.

- Dependencies
Python 3.x

TensorFlow

Keras

Gensim

NLTK

Scikit-learn

Matplotlib

Seaborn


- Installation
Clone the repository:
git clone https://github.com/eyemhaqeeq/Sentiment-Analysis-Using-LSTM-and-Word2Vec-BayesianOpt-.git

Navigate to the project directory:
cd sentiment-lstm

Install dependencies:
pip install -r requirements.txt

Launch the notebook:
jupyter notebook Sentiment_Analysis_LSTM.ipynb

Optional GPU acceleration is recommended for faster training using platforms such as Google Colab or Kaggle.

- Next Steps
Integrate Bayesian Optimization for hyperparameter tuning

Deploy the model using Flask or FastAPI for web-based sentiment prediction

Save/load model weights for reuse

Extend the model to multi-class classification

- License
This project is licensed under the MIT License.

Contact
For questions or collaboration inquiries,
please contact Haqeeq@11gmail.com
