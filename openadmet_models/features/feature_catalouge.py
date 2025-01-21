

FEATURIZER_CLASSES = {}

# decorator to register a featurizer class
def register_featurizer(cls):
    FEATURIZER_CLASSES[cls.type] = cls
    return cls