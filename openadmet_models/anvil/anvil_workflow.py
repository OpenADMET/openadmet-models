from pydantic import BaseModel
from typing import Any
from openadmet_models.util.types import Pathy
from abc import ABC, abstractmethod
import yaml
from loguru import logger


class AnvilWorkflow(BaseModel):
    metadata: Any
    data: Any
    transform: Any
    split: Any
    feat: Any
    model: Any
    eval: Any

    @classmethod
    def from_yaml(cls, path: Pathy):
        """
        Create a workflow from a yaml file
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
        

    def save(self, path: Pathy):
        """
        Save the workflow to a yaml file
        """
        with open(path, "w") as f:
            yaml.dump(self.dict(), f)

    def run(self):
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
        eval = self.eval.evaluate(y_test, preds)
        logger.info("Evaluation done")
        
        return eval

        
