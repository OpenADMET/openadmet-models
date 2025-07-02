# Tutorials
Here are a series of demo notebooks that demonstrate key processes of the OpenADMET package. They are ordered in a typical machine learning workflow.

1. Data Curation  
&nbsp;&nbsp;&nbsp;&nbsp;1.1. Curating external datasets  
&nbsp;&nbsp;&nbsp;&nbsp;1.2. Filtering compounds  

2. Modeling  
&nbsp;&nbsp;&nbsp;&nbsp;2.1. Training models with Anvil  
&nbsp;&nbsp;&nbsp;&nbsp;2.2. Adding a new model type to Anvil  
&nbsp;&nbsp;&nbsp;&nbsp;2.3. Using Weights and Biases  
&nbsp;&nbsp;&nbsp;&nbsp;2.4. Active Learning  

3. Evaluation  
&nbsp;&nbsp;&nbsp;&nbsp;3.1 Comparing models  
---

### 1.1. Curating external datasets
In addition to the provided datasets, you can also plugin datasets from external sources, e.g. your own data or from public data sources. This notebook will walk you through necessary bare minimum data transformation and cleaning that your external data has to undergo to be amenable to modeling.  

### 1.2. Filtering compounds
Depending on your use case, you may be interested in compounds with specific physicochemical properties. E.g. compounds with specific substructural patterns as denoted by SMARTS strings, compounds with specific pKa values. This notebook will walk you through how to do that.  

### 2.1 Training models with Anvil  
Now that you've curated and filtered a dataset, learn how to use Anvil to train an ADMET model! Anvil is our primary infrastructure for model training and evaluation, built to support scalable, reproducible, and rigorous development of ADMET prediction models. Anvil centers around a `yaml`-based recipe system to ensure reproducibility and robustness of model training.

### 2.2 Adding a new model type to Anvil
TBA.  

### 2.3 Using Weights and Biases
TBA.  

### 2.4 Active Learning
TBA.  

### 3.1. Comparing models
Now that you've trained several different models, you can compare their performances across several different metrics to evaluate which model works best for your use case. You can even generate a final comparative report for easy readability.


