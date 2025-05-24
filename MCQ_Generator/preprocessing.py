import nltk
import re
from tqdm import tqdm
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = nltk.word_tokenize(text)
    return [for w in tokens if w not in stop_words]

tqdm.pandas()
df["tokens"] = df["content"].progress_apply(preprocess_text)