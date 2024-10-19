import pandas as pd
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

data = pd.read_csv("data.csv")

data['text'] = data['Title'] + " " + data['Description']
data = data[:500]

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Function to get BERT embeddings for a given text
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the embeddings for the [CLS] token (sentence-level embedding)
    cls_embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embeddings


# Get BERT embeddings for each news article
embeddings = data['text'].apply(get_bert_embedding).tolist()

# Convert embeddings list to a numpy array
embeddings = np.array(embeddings)

# Perform KMeans clustering
num_clusters = 20  # You can choose the number of clusters based on your data
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(embeddings)

# View the clustered data
print(data[['Title', 'Description', 'cluster']])
dicts = data[['Title', 'Description', 'cluster']]
df = pd.DataFrame(data=dicts)
df.to_csv('output.csv', index=False)

