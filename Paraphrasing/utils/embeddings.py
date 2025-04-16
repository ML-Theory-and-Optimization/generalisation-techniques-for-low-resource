from sentence_transformers import SentenceTransformer

def get_sentence_embeddings(texts):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return model.encode(texts, convert_to_tensor=False)