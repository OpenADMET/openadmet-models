

MODEL_CLASSES = {}

# decorator to register a featurizer class
def register_model(cls):
    MODEL_CLASSES[cls.type] = cls
    return cls