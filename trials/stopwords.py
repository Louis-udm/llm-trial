import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as nk_stopwords
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS as sk_stopwords
ENGLISH_STOP_WORDS = set( nk_stopwords.words('english') ).union( set(sk_stopwords) )
print(len(ENGLISH_STOP_WORDS))
print(ENGLISH_STOP_WORDS)
ENGLISH_STOP_WORDS=set.intersection(set(nk_stopwords.words('english')),sk_stopwords)
print(len(ENGLISH_STOP_WORDS))
print(ENGLISH_STOP_WORDS)
