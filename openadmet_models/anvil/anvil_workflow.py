import hashlib
import uuid
from pathlib import Path
from typing import Any

import fsspec
import yaml
from loguru import logger
from pydantic import BaseModel

from openadmet_models.anvil.metadata import Metadata
from openadmet_models.data.data_spec import DataSpec
from openadmet_models.eval.eval_base import EvalBase, get_eval_class
from openadmet_models.features.feature_base import FeaturizerBase, get_featurizer_class
from openadmet_models.models.model_base import ModelBase, get_model_class
from openadmet_models.registries import *  # noqa: F401, F403
from openadmet_models.split.split_base import SplitterBase, get_splitter_class
from openadmet_models.trainer.trainer_base import TrainerBase, get_trainer_class
from openadmet_models.util.types import Pathy

_SECTION_CLASS_GETTERS = {
    "feat": get_featurizer_class,
    "model": get_model_class,
    "split": get_splitter_class,
    "eval": get_eval_class,
    "train": get_trainer_class,
}


def _load_section_from_type(data, section_name, skip_pop=False):
    """
    Load a section from the yaml data
    """
    if skip_pop:
        section_spec = data
    else:
        section_spec = data.pop(section_name)
    section_type = section_spec["type"]
    if "params" in section_spec:
        section_params = section_spec["params"]
    else:
        section_params = {}
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
    trainer: TrainerBase
    evals: list[EvalBase]

    @classmethod
    def from_yaml(cls, path: Pathy, **kwargs):
        """
        Create a workflow from a yaml file
        """
        logger.info("Loading workflow")

        storage_options = kwargs.pop("storage_options", kwargs)
        of = fsspec.open(path, **storage_options)
        with of as stream:
            data = yaml.safe_load(stream)

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

        # trainer
        trainer = _load_section_from_type(data, "train")

        # load the evaluations we want to do
        evals = []
        eval_spec = data.pop("eval")
        for eval_subspec in eval_spec:
            eval_instance = _load_section_from_type(eval_subspec, "eval", skip_pop=True)
            evals.append(eval_instance)

        # make the complete instance
        instance = cls(
            metadata=metadata,
            data_spec=data_spec,
            model=model,
            feat=featurizer,
            evals=evals,
            split=split,
            trainer=trainer,
            **data,
        )

        logger.info("Workflow loaded")

        return instance

    def save(self, path: Pathy):
        """
        Save the workflow to a yaml file
        """
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Pathy):
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def run(self, output_dir: Pathy = "anvil_run") -> Any:
        """
        Run the workflow
        """
        output_dir = str(output_dir)
        if Path(output_dir).exists():
            # make truncated hashed uuid
            hsh = hashlib.sha1(str(uuid.uuid4()).encode("utf8")).hexdigest()[:8]
            output_dir = Path(output_dir + f"_{hsh}")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running workflow from directory {output_dir}")

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
        X_train_feat, _  = self.feat.featurize(X_train)
        X_test_feat, _ = self.feat.featurize(X_test)
        logger.info("Data featurized")


        logger.info("Building model")
        self.model.build()
        logger.info("Model built")

        logger.info("Setting model in trainer")
        self.trainer.model = self.model
        logger.info("Model set in trainer")

        logger.info("Training model")
        self.model = self.trainer.train(X_train_feat, y_train)
        logger.info("Model trained")

        logger.info("Saving model")
        self.model.to_model_json_and_pkl(
            output_dir / "model.json", output_dir / "model.pkl"
        )
        logger.info("Model saved")

        logger.info("Predicting")
        preds = self.model.predict(X_test_feat)
        logger.info("Predictions made")

        logger.info("Evaluating")
        for eval in self.evals:
            eval.evaluate(y_test, preds)
            eval.report(write=True, output_dir=output_dir)
        logger.info("Evaluation done")
