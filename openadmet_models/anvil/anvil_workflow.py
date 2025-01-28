from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel

from openadmet_models.anvil.metadata import Metadata
from openadmet_models.data.data_spec import DataSpec
from openadmet_models.eval.eval_base import (EvalBase, evaluators,
                                             get_eval_class)
from openadmet_models.features.feature_base import (FeaturizerBase,
                                                    featurizers,
                                                    get_featurizer_class)
from openadmet_models.models.model_base import (ModelBase, get_model_class,
                                                models)
from openadmet_models.split.split_base import (SplitterBase,
                                               get_splitter_class, splitters)
from openadmet_models.util.types import Pathy
import fsspec

_SECTION_CLASS_GETTERS = {
    "feat": get_featurizer_class,
    "model": get_model_class,
    "split": get_splitter_class,
    "eval": get_eval_class,
}


def _load_section_from_type(data, section_name):
    """
    Load a section from the yaml data
    """
    section_spec = data.pop(section_name)
    section_type = section_spec["type"]
    section_params = section_spec["params"]
    section_class = _SECTION_CLASS_GETTERS[section_name](section_type)
    section_instance = section_class(**section_params)
    return section_instance


class AnvilWorkflow(BaseModel):
    metadata: Metadata
    data_spec: DataSpec
    transform: Any
    split: SplitterBase
    feat: FeaturizerBase
    model: ModelBase
    evals: list[EvalBase]
    _ANVIL_DIR: Pathy 

    @classmethod
    def from_yaml(cls, path: Pathy, **kwargs):
        """
        Create a workflow from a yaml file
        """
        logger.info("Loading workflow")

        storage_options = kwargs.pop("storage_options", kwargs)
        of = fsspec.open(path, **storage_options)
        with of as stream:
            data=yaml.safe_load(stream)

        # get parent path
        parent = of.fs.unstrip_protocol(of.fs._parent(path))

        data_spec = DataSpec(**data.pop("data"), anvil_dir=parent)

        metadata = Metadata(**data.pop("metadata"))

        # load the featurizer(s)
        featurizer = _load_section_from_type(data, "feat")

        # model
        model = _load_section_from_type(data, "model")

        # split
        split = _load_section_from_type(data, "split")

        # load the evaluations we want to do
        evals = []
        eval_spec = data.pop("eval")
        for eval_type in eval_spec:
            eval_class = get_eval_class(eval_type)
            evals.append(eval_class())

        # make the complete instance
        instance = cls(
            metadata=metadata,
            data_spec=data_spec,
            model=model,
            feat=featurizer,
            evals=evals,
            split=split,
            **data,
        )

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
        X, y = self.data_spec.read()
        logger.info("Data loaded")

        logger.info("Transforming data")
        if self.transform:
            X = self.transform.transform(X)
            logger.info("Data transformed")
        else:
            logger.info("No transform specified, skipping")

        logger.info("Splitting data")
        X_train, X_test, y_train, y_test = self.split.split(X, y)
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
