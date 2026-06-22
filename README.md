# BM-NLP
Source code for the paper "Natural Language Processing Based Solution for Labeling Brain Metastasis Identified in Radiology Reports"

# Documentation for Model Evaluation Pipeline

The packages in our environment is listed in `requirements.txt`. 

Python version: 3.11.3

Install the packages with pip:
>$ python3 -m venv env \
>$ source env/bin/activate\
>$ pip install -r requirements.txt

Or with conda:
>$ conda create --name <env> --file <this file>

## Data Processing
`data_utils.py` contains regex functions for data cleaning that are used in `data_processing.py`. `data_processing.py` separates `findings` and `impression` sections from the report texts and counts the number of tokens for each section. It keeps reports with both sections and with impressions shorter than 512 tokens. Note that the context length of the model is 512 tokens, text longer than this will be trimmed to fit.

Modify code in `data_processing.py` to adapt to your data structure:
- Read you dataset into a Pandas DataFrame
- Decide the information you want to keep. Unique identifiers are useful for analysis. For the Ontario data, `anon_accessions` and `anon_pids` are both needed to uniquely identify a report.
- If the dataset is unlabeled, call `organize_unlabeled_data(df)`. Otherwise, call `organize_met_labeled_data(df)`
- Save the data in `*.pkl` format.
## Model Inference
The code for inference and evaluation are all contained in `inference.py`.

Explain the arguments:
- `--labeled` should be set if your dataset is labeled. This will trigger the model evaluation code. Do not set it if your data is unlabeled.
- `view1_name` and `view2_name` are used to separate models and dataloaders used for `findings` and `impression` sections of the report. Keep it as default.
- `--target` is the column name for the label. This is useful for evaluation phase.
- `--threshold` is the model prediction threshold. `probability > threshold` makes the prediction result 1.
- `--recall-thresh` is set to 0.9 since we want to see which threshold can achieve 0.9 sensitivity. This is for the evaluation phase.
- `--test-data-pkl` is the data pkl file for inference or evaluation.
- `--weight-path` should be the path to a csv including the weight for each sample. It should contain two columns, `ID` (or other identifiers) and `met_label_weight`. This can be optionally used for evaluation. Keep it as empty if unweighted evaluation is needed.

Command:
> python inference --test-data-pkl data/your_processed_reports.pkl 

After running the command, a file named `met_label_predictions.csv` will be produced. It contains 4 columns:
- `predicted`: The predicted BM label
- `probabilities`: The ensembled (average) probability
- `findings_prob`: Probability predicted on the findings section
- `impressions_prob`: Probability predicted on impressions section
## Test Data Sampling
`plot_bins.ipynb` plots the bins and the population each bins. 
`sample_reports.py` provides an example of stratified sampling by predicted probabilities. Change `samples_per_bin` based on your data distribution.

Annotate your sampled data by *yes, no, or possible* and save it in `*.pkl` format. Note that the label column name should be the same as `--target` in `inference.py`.

## Model Evaluation

Command:
> python inference --labeled --test-data-pkl data/your_labeled_samples.pkl --threshold 0.4 
This will generate the model predicted probabilities. Then refer to   analyse_ir2.ipynb` to calculate N, n , y1, yhat1, tp
 -  Total number of reports/pts per bin
 -  Sampled reports/pts per bin
 -  # human labels Y=1 in the sampled subset
 -  # model predicted=1 in the sampled subset
 -  # TP (Y=1 AND predicted=1) in the sampled subset

Calculate the precision, recall, and F1 using `precision_recall_calc.ipynb`.
