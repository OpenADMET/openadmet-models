Example Tutorial Notebooks
==========================

We recommend starting with our interactive tutorials hosted as a
`web-book <https://demos.openadmet.org>`_.
These tutorials introduce the fundamentals of using **OpenADMET Models** and provide hands-on, end-to-end examples.
The corresponding source code and materials are available on
`GitHub <https://github.com/OpenADMET/openadmet-demos>`_.

---

Overview
--------

In these tutorials, we will walk through a typical **ADMET modeling** workflow — from data curation to model training, comparison, and inference.

As a case study, we focus on **Cytochrome P450 (CYP450) inhibition**, specifically **CYP3A4**, the most abundant hepatic isoform and the enzyme responsible for metabolizing nearly 50% of marketed drugs.
CYP3A4 inhibition is a major driver of **drug–drug interactions (DDIs)** and is therefore a critical endpoint in early-stage drug discovery.

---

What is CYP3A4 inhibition and how is it measured?
-------------------------------------------------

CYP3A4 inhibition occurs when a compound decreases the enzymatic activity of CYP3A4, slowing or blocking substrate metabolism. This can occur through:

* **Reversible inhibition:** The inhibitor binds transiently, and normal activity resumes upon dissociation.
* **Irreversible inhibition:** The inhibitor permanently inactivates the enzyme, requiring new CYP3A4 synthesis to restore activity.

Inhibition is typically measured **in vitro** using enzyme assays with probe substrates.
The most common metric is the :math:`IC_{50}` — the concentration of inhibitor required to reduce enzyme activity by 50%.
Lower :math:`IC_{50}` values indicate stronger inhibition.

---

Tutorial Structure
------------------

All tutorial notebooks are available in the ``demos/`` directory of the GitHub repository.
Each notebook demonstrates a key step in building and deploying a CYP3A4 inhibition model using OpenADMET tooling.

1. `Curating CYP3A4 data from ChEMBL <https://demos.openadmet.org/en/latest/demos/01_Data_Curation/01_Curate_ChEMBL_Data.html>`_
   Retrieve relevant CYP3A4 inhibition data from public sources (e.g., ChEMBL) and perform essential cleaning and preprocessing to prepare data for modeling.

2. `Training CYP3A4 models with Anvil <https://demos.openadmet.org/en/latest/demos/02_Model_Training/02_Training_Models.html>`_
   Train machine learning models using **Anvil**, OpenADMET’s YAML-based infrastructure for scalable and reproducible model development.

3. `Comparing trained models <https://demos.openadmet.org/en/latest/demos/03_Model_Comparison/03_Comparing_Models.html>`_
   Evaluate and compare model performance across metrics, and generate standardized reports. Learn how to incorporate your own (BYO) models for comparison.

4. `Training a CYP3A4 model ensemble <https://demos.openadmet.org/en/latest/demos/04_Ensemble_Model_Training/04_Ensemble_Model_Training_Active_Learning.html>`_
   Use the best-performing models to create an ensemble. Ensembles provide uncertainty estimates, helping contextualize predictions and improve decision-making.

5. `Running model ensemble inference <https://demos.openadmet.org/en/latest/demos/05_Ensemble_Model_Inference/05_Model_Ensemble_Inference.html>`_
   Apply trained ensembles to predict CYP3A4 inhibition on unseen datasets, such as lead series or screening compounds.
   Introduces active learning workflows for prioritizing compounds for testing.


---

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: Curating CYP3A4 Inhibition Data from ChEMBL
      :margin: 0 3 0 0
      :text-align: center
      :link: https://demos.openadmet.org/en/latest/demos/01_Data_Curation/01_Curate_ChEMBL_Data.html
      :link-type: url

        Learn how to retrieve and preprocess CYP3A4 inhibition data from ChEMBL for downstream modeling.

    .. grid-item-card:: Training Models
      :margin: 0 3 0 0
      :text-align: center
      :link: https://demos.openadmet.org/en/latest/demos/02_Model_Training/02_Training_Models.html
      :link-type: url

        Follow a step-by-step guide to train machine learning models using OpenADMET and Anvil.

    .. grid-item-card:: Comparing Models
      :margin: 0 3 0 0
      :text-align: center
      :link: https://demos.openadmet.org/en/latest/demos/03_Model_Comparison/03_Comparing_Models.html
      :link-type: url

        Learn how to compare model performance, visualize results, and benchmark across approaches.

    .. grid-item-card:: Ensemble Model Training
      :margin: 0 3 0 0
      :text-align: center
      :link: https://demos.openadmet.org/en/latest/demos/04_Ensemble_Model_Training/04_Ensemble_Model_Training_Active_Learning.html
      :link-type: url

        Build ensemble models and integrate active learning strategies to quantify uncertainty and improve prediction robustness.

    .. grid-item-card:: Model Inference
      :margin: 0 3 0 0
      :text-align: center
      :link: https://demos.openadmet.org/en/latest/demos/05_Ensemble_Model_Inference/05_Model_Ensemble_Inference.html
      :link-type: url

        Run ensemble inference on new datasets and interpret CYP3A4 inhibition predictions.

    .. grid-item-card:: Showcase Notebook
      :margin: 0 3 0 0
      :text-align: center
      :link: https://try.openadmet.org
      :link-type: url

        Explore a comprehensive, end-to-end showcase notebook hosted on Binder — ideal for a quick overview of OpenADMET Models in action.
