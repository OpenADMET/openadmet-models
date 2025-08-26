from typing import ClassVar, Literal, Optional, Union

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor, AutoTabPFNClassifier
from tabpfn import TabPFNRegressor, TabPFNClassifier
import numpy as np
from loguru import logger
from pydantic import field_validator, Field

from openadmet.models.architecture.model_base import PickleableModelBase, models



class TabPFNExtensionModelBase(PickleableModelBase):
    """
    Base class for TabPFN models.
    """

    type: ClassVar[str]
    mod_class: ClassVar[type]  # To specify the TabPFN model class (e.g., AutoTabPFNRegressor or AutoTabPFNClassifier)

    max_time: Optional[int] = Field(
        default=None,
        description="The maximum time to spend on fitting the post hoc ensemble."
    )

    accelerator: Literal["cpu", "gpu", "auto"] = Field(
        default="auto",
        description="The device to use for training and prediction."
    )  # TABPFN calls this "device" but we use "accelerator" here to match Pytorch
       # tabpfn doesn't use the same device convention as pytorch (cuda vs gpu), we unify here

    random_state: int = Field(
        default=42,
        description="Controls both the randomness of base models and the post hoc ensembling method."
    )


    ignore_pretraining_limits: bool = Field(
        default=False,
        description="Whether to ignore the pretraining limits of the TabPFN base models."
    )

    phe_init_args: Optional[dict] = Field(
        default=None,
        description="The initialization arguments for the post hoc ensemble predictor. "
                    "See post_hoc_ensembles.pfn_phe.AutoPostHocEnsemblePredictor for more options and all details."
    )



    @field_validator("accelerator")
    @classmethod
    def validate_accelerator(cls, value):
        """
        Validate the accelerator parameter
        """
        if value not in ["cpu", "gpu", "auto"]:
            raise ValueError("Accelerator must be either 'cpu' or 'gpu' or 'auto'")

        return value


    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters
        """

        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance


    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model
        """
        self.build()
        self.estimator = self.estimator.fit(X, y)

    def build(self):
        """
        Prepare the model
        """
        # tabpfn doesn't use the same device convention as pytorch, we unify here
        accelerator = self.accelerator if self.accelerator != "gpu" else "cuda"
        if not self.estimator:
            self.estimator = self.mod_class(max_time=self.max_time,
                                            device=accelerator,
                                            random_state=self.random_state,
                                            ignore_pretraining_limits=self.ignore_pretraining_limits,
                                            phe_init_args=self.phe_init_args)
        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("TabPFNPostHocRegressorModel")
class TabPFNPostHocRegressorModel(TabPFNExtensionModelBase):
    """
    TabPFN regression model using `tabpfn-extensions`, posthoc ensembling.
    """

    type: ClassVar[str] = "TabPFNPostHocRegressorModel"
    mod_class: ClassVar[type] = AutoTabPFNRegressor


@models.register("TabPFNPostHocClassifierModel")
class TabPFNPostHocClassifierModel(TabPFNExtensionModelBase):
    """
    TabPFN classification model using `tabpfn-extensions`, posthoc ensembling.
    """

    type: ClassVar[str] = "TabPFNPostHocClassifierModel"
    mod_class: ClassVar[type] = AutoTabPFNClassifier


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)



class TabPFNModelBase(PickleableModelBase):
    accelerator: Literal["cpu", "cuda", "auto"] = Field(default="auto")
    random_state: int = Field(default=42)
    ignore_pretraining_limits: bool = Field(default=False)

    @classmethod
    def from_params(cls, class_params: dict = {}, mod_params: dict = {}):
        """
        Create a model from parameters
        """

        instance = cls(**class_params, mod_params=mod_params)
        instance.build()
        return instance

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model
        """
        self.build()
        self.estimator = self.estimator.fit(X, y)

    def build(self):
        """
        Prepare the model
        """
        accelerator = self.accelerator if self.accelerator != "gpu" else "cuda"
        # tabpfn doesn't use the same device convention as pytorch (cuda vs gpu), we unify here
        if not self.estimator:
            self.estimator = self.mod_class(device=accelerator, # TABPFN calls this device
                                            random_state=self.random_state,
                                            ignore_pretraining_limits=self.ignore_pretraining_limits)
        else:
            logger.warning("Model already exists, skipping build")

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Predict using the model
        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return np.expand_dims(self.estimator.predict(X), axis=1)


@models.register("TabPFNRegressorModel")
class TabPFNRegressorModel(TabPFNModelBase):
    """
    TabPFN regression model using the basic `tabpfn` implementation.
    """

    type: ClassVar[str] = "TabPFNRegressorModel"
    mod_class: ClassVar[type] = TabPFNRegressor


@models.register("TabPFNClassifierModel")
class TabPFNClassifierModel(TabPFNModelBase):
    """
    TabPFN classification model using the basic `tabpfn` implementation.
    """

    type: ClassVar[str] = "TabPFNClassifierModel"
    mod_class: ClassVar[type] = TabPFNClassifier
