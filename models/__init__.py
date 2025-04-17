def get_model(name, embedding_matrix, args):
    name = name.lower()
    if name == "baseline":
        from .baseline import BaselineClassifier
        return BaselineClassifier(embedding_matrix, args)
    elif name == "lstm":
        from .lstm import LSTMClassifier
        return LSTMClassifier(embedding_matrix, args)
    elif name == "bilstm":
        from .bilstm import BiLSTMClassifier
        return BiLSTMClassifier(embedding_matrix, args)
    elif name == "bilstm_max":
        from .bilstm_max import BiLSTMMaxPoolClassifier
        return BiLSTMMaxPoolClassifier(embedding_matrix, args)
    else:
        raise ValueError(f"Unknown model type: {name}")