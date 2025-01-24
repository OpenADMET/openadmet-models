

EVAL_CLASSES = {}

# decorator to register a featurizer class
def register_eval(cls):
    EVAL_CLASSES[cls.type] = cls
    return cls


def get_eval_class(eval_type: str):
    try:
        eval_class = EVAL_CLASSES[eval_type]
    except KeyError:
        raise ValueError(f"Featurizer type {eval_type} not found in featurizer catalouge")
    
    return eval_class