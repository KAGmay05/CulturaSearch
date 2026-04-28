from neural_based_model.neural_retriever import NeuralRetriever


class RetrieverWrapped:
    
    def __init__(self):
        self.retriever = NeuralRetriever()

    def search(self, query, top_k=3):
        return self.retriever.search(query, top_k)    

    def search_with_web_expansion(self, query, top_k=3):
        return self.retriever.search_with_web_expansion(query, top_k=top_k)