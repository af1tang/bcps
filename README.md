# Adversarial Precision Sensing w/ Healthcare Applications

This is the code repository for *Adversarial Precision Sensing with Healthcare Applications* (ICDM'20) by Fengyi Tang, Lifan Zeng, Fei Wang and Jiayu Zhou. 
Here, you will find instructions for:

* preparing MIMIC-III code to test out experiments
* preparing synthetic data from the paper
* `models.py` file containing the FS and PL models
* `main.py` training and evaluation loops

## Usages
### Requirements
* Python 3.4+
* Pytorch (https://pytorch.org/)
* Scikit-Learn
* Gensim
* NumPy
* Matplotlib
* Seaborn
* Pandas
* Tensorflow 1.11+
* Progressbar2
* Postgres (or equivalent for building local MIMIC-III)

### MIMIC-III ###
Please apply for access to the publicly available MIMIC-III DataBase via `https://www.physionet.org/`. 

### Files ###
* `utils.py`: utilities file containing visualization methods, variable conversion to GPU compatilibity form, subsampling, ... etc.
* `models.py`: contains all the model classes for inference models as well as the FS and PL. 
* `preprocessing.py`: preprocesses local MIMIC-III data into `X` and `y`.
* `main.py`: runs the main training loop, testing, and trade-off studies.  

### Instructions for Use ###

**Workflow**: MIMIC-III Access -> Obtain Views and Tables -> Preprocessing -> Pipeline

1. Obtain access to MIMIC-III and clone this repo to local folder. 
Create a local MIMIC-III folder to store a few files:
* `.../local_mimic`
* `.../local_mimic/views`
* `.../local_mimic/tables`
* `.../local_mimic/save`

These paths will be important for storing views and pivot tables, which will be used for preprocessing.

2. Build MIMIC-III database using `postgres`, follow the instructions outlined in the MIMIC-III repository: 
`https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic/postgres`.

3. Go to the pivot folder in the MIMIC-III repository:
`https://github.com/MIT-LCP/mimic-code/tree/master/concepts/pivot`.
Run use the `.sql` scripts to build a local set of `.csv` files of the pivot tables:
* pivoted-bg.sql 
* pivoted_vital.sql
* pivoted_lab.sql
* pivoted_gcs.sql (optional)
* pivoted_uo.sql (optional)

When running the `.sql` script, change the _delimiter_ of the materialized views to `','` when saving as `.csv` file.  

For example,  
`mimic=> \copy (select * FROM mimiciii.icustay_detail) to 'icustay_detail.csv' delimiter ',' csv header;`

After running these scripts, you should have obtained local `.csv` files of the pivot tables. 
Create a local folder to place them in, i.e. `.../local_mimic/views/pivoted-bg.csv`. 
Remember this `.../local_mimic/views` folder, as it will be the `path_views` input for preprocessing purposes.

4. Go to the demographics folder in the MIMIC-III repository:
`https://github.com/MIT-LCP/mimic-code/tree/master/concepts/demographics`.

Run `icustay-detail.sql` and obtain a local `.csv` file of `icustays-detail` view. 
Create a local folder to place the `.csv` file in, i.e.`.../local_mimic/views/icustay_details.csv`. 
Again, have this `.csv` file inside the local `views` folder.

A minor change needs to be made in `icustay_details.csv`:  
change `'admission_age' -> 'age'` for the column header in the `.csv` file manually. 

5. Obtain a local copy of the following tables from MIMIC-III:
* admissions.csv
* diagnoses_icd.csv
* d_icd_diagnoses.csv

These can be directly obtained from *Physionet* as compressed files. 
While tables such as `chartevents` are large, the above tables are quite small and easy to query directly if a local copy is available. 

Save these tables under `.../local_mimic/tables` folder. 
Make the following changes: 
* In `~/local_mimic/tables/diagnoses_icd.csv`, change the column titles `"ROW_ID","SUBJECT_ID","HADM_ID","SEQ_NUM","ICD9_CODE"` to
`"row_id","subject_id","hadm_id","seq_num","icd9_code"` (i.e., make lower case). 
* In `~local_mimic/tables/d_icd_diagnoses.csv` change the column titles `"ROW_ID","ICD9_CODE","SHORT_TITLE","LONG_TITLE"` to
`"row_id","icd9_code","short_title","long_title"` (i.e., again, make lower case). 

6. Run `preprocessing.py` with inputs: 
* `--path_tables <path_tables>`
* `--path_views <path_views>`
* `--path_save <path_save>`.

`<path_tables>` and `<path_views>`  should correspond to the folders under which the local tables and views (pivots and icustays-details) are saved.
 `<path_save>` corresponds to the desired folder to save your variables for training and beyond.
 
 `preprocessing.py` will generate the following files:
 * `X.npy`: main feature tensor, consisting of time-series data generated from a combination of 19 lab values and vital signs over 48 hour period from start of admissions. 
 * `labels`:  raw file containing `hadm_id` (hospital admissions identifier) and labels of interest. 
 * `y`: main label matrix with mortality label for each patient.   
 
7. Run `main.py` with selection of task, training, and testing conditions:
* `--features_dir`: path to saved the feature file to use as X. Selections include `X19`, `X48`, `sentences`, or `onehot`.
* `--y_dir`: path to `y`.
* `--task`: specifies the learning task. User can choose between `['mort', 'sep']` (default = `mort`).
* `--budget`: specifies budget to be used for main training and testing loops (default=1e-5).
* `--checkpoint_dir`: specifies the path to save best models and testing results. 
* `--hidden_size`: specifies number of hidden units for deep models (default=128). 
* `--l1_reg`: regularization for inference models (default=5e-6)
* `--learning_rate`: specifies the initial learning rate (default=0.005).
* `--training_epochs`: number of training epochs for inference models (default=45).
* `--num_epochs`: max number of training epochs for FS and PL (default=30).
*`--batch_size`: batch size during training (default = 32).

The main program runs the following items:
 + 5 evaluation runs of the inference models on the sensed vs. original data.
 + Sparsity trade-off experiments.
The default `beta` is set to `5e-6`, which yields the best results but can be unstable during training. 
The main function returns `stats` and `beta_data` for each of the experiments and saves `stats` (main experiments). 
Models are saved under `checkpoint_dir` provided by the user.

8. Instructions for running Synthetic Experiments.
* First, download the dataset `synthetic.data.gz` from https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic-mld/.
* Unzip the `synthetic.data.gz` to obtain `synthetic.data` file. Save this file under some desired path, i.e, `filename`.
* Run `syn_exp(filename)`, with `filename` being the path to the saved `.data` file.
* `syn_exp` returns: 
	+ `data`: generated synthetic (X,y) pairs used for the experiments. 
	+ `stats`: dictionary of the form {'data': performance, 'budget': % sparsity in `C`, 'G': parameters of FS, 'D': parameters of PL}
	+ `mask`: BCPS sensed features.
	+ `ground_truth`: true `C`. 

In this version, one has to _manually_ change the sparsity patterns in the ground truth `C`. 
Go to `X[:, xx:yy, zz:aa]` and `ground_truth[:, xx:yy, zz:aa]` and change the `xx:yy` to the desired time-steps and `zz:aa` to be the desired features that contribute to the underlying `y` label. 
Similarly, in the `y` label loop, change `if (np.sum(X[i][xx:yy, zz:aa]) + np.sum(...) ... > 0)` to the same set of `xx:yy` and `zz:aa` for `ground_truth`. 

To visualize the sensed features, use `visualize_mask(mask[i])` for the _i-th_ test sample. 

### References ###

If you find this repository helpful for your work, please consider citing us.
<!-- 
```
@article{Tang2020AdversarialPS,
	title={Adversarial Precision Sensing with Healthcare Applications},
	author={Fengyi Tang and Lifan Zeng and Fei Wang and Jiayu Zhou},
	journal={IEEE International Conference on Data Mining (ICDM)},
  	year={2020},
  	pages={000-999}

}
``` -->