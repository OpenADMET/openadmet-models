

MODEL_CLASSES = {}

# decorator to register a featurizer class
def register_model(cls):
    MODEL_CLASSES[cls.type] = cls
    return cls


def get_model_class(model_type: str):
    try:
        model_class = MODEL_CLASSES[model_type]
    except KeyError:
        raise ValueError(f"Model type {model_type} not found in model catalouge")
    
    return model_class