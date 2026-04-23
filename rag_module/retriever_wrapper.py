from neural_based_model.neural_retriever import NeuralRetriever


class RetrieverWrapped:
    """Wrapper alrededor de NeuralRetriever para compatibilidad."""
    
    def __init__(self):
        self.retriever = NeuralRetriever()

    def search(self, query, top_k=3):
        """Buscar documentos relevantes."""
        return self.retriever.search(query, top_k)    