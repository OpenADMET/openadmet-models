

FEATURIZER_CLASSES = {}

# decorator to register a featurizer class
def register_featurizer(cls):
    FEATURIZER_CLASSES[cls.type] = cls
    return cls


def get_featurizer_class(featurizer_type: str):
    try:
        featurizer_class = FEATURIZER_CLASSES[featurizer_type]
    except KeyError:
        raise ValueError(f"Featurizer type {featurizer_type} not found in featurizer catalouge")
    
    return featurizer_class