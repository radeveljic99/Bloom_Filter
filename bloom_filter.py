import mmh3
from bitarray import bitarray

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_document(document):
    
    tokens = nltk.word_tokenize(document)

    tokens = [token.lower() for token in tokens]

    tokens = [token for token in tokens if token not in string.punctuation]

    tokens = [token for token in tokens if not token.isdigit()]

    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    preprocessed_document = ' '.join(tokens)

    return preprocessed_document


def preprocess_documents(documents):
    
    vectorizer = CountVectorizer(min_df=3)

    preprocessed_documents = [preprocess_document(doc) for doc in documents]

    vectorized_documents = vectorizer.fit_transform(preprocessed_documents)

    frequent_words = vectorizer.get_feature_names_out()

    return frequent_words

class BloomFilter:
    def __init__(self, capacity, num_hashes):
        self.capacity = capacity
        self.num_hashes = num_hashes
        self.bit_array = bitarray(capacity)
        self.bit_array.setall(0)

    def add(self, item):
        for seed in range(self.num_hashes):
            index = mmh3.hash(item, seed) % self.capacity
            self.bit_array[index] = 1

    def contains(self, item):
        for seed in range(self.num_hashes):
            index = mmh3.hash(item, seed) % self.capacity
            if self.bit_array[index] == 0:
                return False
        return True




def check_outliers(bloom_filter, known_documents, new_document, similarity_threshold, frequent_words):
    
    if bloom_filter.contains(new_document):
        return False
    
    for document in known_documents:
        jaccard_similarity = calculate_jaccard_similarity(document, new_document, frequent_words)
        if jaccard_similarity >= similarity_threshold:
            return False
    return True


def calculate_jaccard_similarity(doc1, doc2, frequent_words):
    set1 = set(doc1.split()).intersection(set(frequent_words))
    set2 = set(doc2.split()).intersection(set(frequent_words))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union if union != 0 else 0
    return similarity




bloom_filter = BloomFilter(10000, 3)

documents_to_add = ['data/sport_' + str(i) + '.txt' for i in range(1, 80)]
    
documents_to_add_content = []

documents_to_test = ['data/sport_' + str(i) + '.txt' for i in range(70, 90)]


for document in documents_to_add:
    with open(document, "r") as file:
        document_text = file.read()
        documents_to_add_content.append(preprocess_document(document_text))
        bloom_filter.add(preprocess_document(document_text))
        
        
frequent_words = preprocess_documents(documents_to_add_content)
# print(frequent_words)    

similarity_threshold = 0.1

for document in documents_to_test:
    with open(document, "r") as file:
        document_text = file.read()
        if  check_outliers(bloom_filter, documents_to_add_content, preprocess_document(document_text), similarity_threshold, frequent_words):
            print(f"Dokument '{document}' vjerovatno jeste outlier")
        else:
            print(f"Dokument '{document}' vjerovatno nije outlier")
