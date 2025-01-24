from pydantic import BaseModel
from typing import Any
import yaml
from loguru import logger


from openadmet_models.util.types import Pathy
from openadmet_models.anvil.metadata import Metadata
from openadmet_models.models.model_base import ModelBase, get_model_class
from openadmet_models.features.feature_base import FeaturizerBase, get_featurizer_class
from openadmet_models.eval.eval_base import EvalBase, get_eval_class



class AnvilWorkflow(BaseModel):
    metadata: Metadata
    data: Any
    transform: Any
    split: Any
    feat: FeaturizerBase
    model: ModelBase
    evals: list[EvalBase]

    @classmethod
    def from_yaml(cls, path: Pathy):
        """
        Create a workflow from a yaml file
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        metadata = Metadata(**data.pop("metadata"))

        # load the featurizer(s)
        featurizer_spec = data.pop("feat")
        featurizer_type = featurizer_spec["type"]
        featurizer_params = featurizer_spec["featurizer_params"]
        featurizer_class = get_featurizer_class(featurizer_type)
        featurizer = featurizer_class(**featurizer_params)



        # load the model
        model_spec = data.pop("model")
        model_type = model_spec["type"]
        model_params = model_spec["model_params"]
        model_class = get_model_class(model_type)
        model = model_class.from_params(model_params=model_params)


        # load the evaluations we want to do
        evals = []
        eval_spec = data.pop("eval")
        for eval_type in eval_spec:
            eval_class = get_eval_class(eval_type)
            evals.append(eval_class())


        # make the complete instance
        instance = cls(metadata=metadata, model=model, feat=featurizer, evals=evals, **data)
        
        logger.info("Workflow loaded")

        return instance

        

    def save(self, path: Pathy):
        """
        Save the workflow to a yaml file
        """
        with open(path, "w") as f:
            yaml.dump(self.dict(), f)

    def run(self) -> Any:
        """
        Run the workflow
        """
        logger.info("Running workflow")
        
        logger.info("Loading data")
        X = self.data.load()
        logger.info("Data loaded")

        logger.info("Transforming data")
        transformed_X = self.transform.transform(X) # can be no-op
        logger.info("Data transformed")

        logger.info("Splitting data")
        X_train, X_test, y_train, y_test = self.split.split(transformed_X)
        logger.info("Data split")

        logger.info("Featurizing data")
        X_train_feat = self.feat.featurize(X_train)
        logger.info("Data featurized")

        logger.info("Training model")
        model = self.model.load()
        model.train(X_train_feat, y_train)
        logger.info("Model trained")

        logger.info("Predicting")
        preds = model.predict(X_test)
        logger.info("Predictions made")
        
        logger.info("Evaluating")
        report_data = [eval.evaluate(y_test, preds) for eval in self.evals]
        logger.info("Evaluation done")
        
        return report_data

        
