# SENTIMENT-ANALYSIS

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: AYESHA UMRA TABASSUM

*INTERN ID*: CITS589

*DOMAIN*: Data Analytics

*DURATION*: 4 Weeks

*MENTOR*: NEELA SANTOSH

# TASK-4: SENTIMENT ANALYSIS USING NATURAL LANGUAGE PROCESSING (NLP)

## Objective

The objective of this project is to perform Sentiment Analysis on textual data using Natural Language Processing (NLP) techniques. Sentiment Analysis is a text mining process used to identify and classify opinions expressed in text as positive, negative, or neutral. In this project, the IMDb Movie Reviews dataset was used to analyze customer opinions and automatically determine the sentiment associated with each review.

## Tools and Technologies Used

* Google Colab
* Python
* Pandas
* NumPy
* NLTK (Natural Language Toolkit)
* Scikit-learn
* Matplotlib
* Seaborn

Google Colab was selected as the development environment because it provides a cloud-based platform for executing Python code without requiring local installation of software packages.

## Methodology

The project began with loading the IMDb Movie Reviews dataset into the Colab environment. The dataset contains movie reviews labeled as either positive or negative. Data preprocessing was performed to improve the quality of textual information before model training.

The preprocessing steps included converting text to lowercase, removing HTML tags, eliminating special characters, removing stopwords, and cleaning unnecessary text elements. These steps helped reduce noise and improve the effectiveness of the machine learning model.

After preprocessing, the text data was transformed into numerical form using TF-IDF (Term Frequency–Inverse Document Frequency) Vectorization. This technique converts text into meaningful numerical features that machine learning algorithms can process.

The dataset was then divided into training and testing sets using an 80:20 ratio. A Logistic Regression classifier was trained on the processed data to predict the sentiment of movie reviews. Model performance was evaluated using accuracy score, classification report, and confusion matrix.

## Results

The sentiment classification model successfully identified positive and negative reviews with high accuracy. The confusion matrix demonstrated effective classification performance, while the classification report provided detailed evaluation metrics such as precision, recall, and F1-score.

The trained model was also tested using custom review inputs, and it accurately predicted the sentiment associated with the provided text.

## Insights

1. Sentiment Analysis successfully classified movie reviews into positive and negative categories.
2. Data preprocessing significantly improved text quality and model performance.
3. TF-IDF vectorization effectively transformed textual information into numerical features.
4. Logistic Regression proved to be a reliable algorithm for sentiment classification tasks.
5. The model achieved strong predictive performance on unseen test data.
6. NLP techniques enabled efficient extraction of valuable insights from large volumes of textual data.
7. Sentiment Analysis can be applied to customer reviews, social media comments, product feedback, and opinion mining applications.

## Conclusion

This project demonstrated the practical implementation of Natural Language Processing and Machine Learning for sentiment classification. By combining text preprocessing, feature extraction, and predictive modeling, meaningful insights were obtained from unstructured textual data. The project highlights the importance of NLP in understanding customer opinions and supporting data-driven decision-making.

## OUTPUT

<img width="1078" height="623" alt="Image" src="https://github.com/user-attachments/assets/96b50812-5bc9-4f87-854f-15b1901c1e5d" />
