This repository provides the Pytorch implementations for NeurIPS 2024 paper "Leveraging Tumor Heterogeneity: Heterogeneous Graph Representation Learning for Cancer Survival Prediction in Whole Slide Images".

Paper can be found [here](https://proceedings.neurips.cc/paper_files/paper/2024/hash/760341adc5632de3f1cf2e8d22215a93-Abstract-Conference.html).
# Data Preparation
## Download the WSIs
The WSIs can be found in [TCGA](https://www.cancer.gov/tcga).

## Patch Extraction
We follow [CLAM](https://github.com/mahmoodlab/CLAM) to cut whole-slide images (WSIs) into patches (size $256\times 256$ at $20\times$ magnification),
and then generate the instance-level features using the UNI encoder.
In the subsequent steps, we follow CLAM's storage format to obtain the patch coordinates and features.

## Patch Classification
Our initial patch classifier was obtained through fine-tuning [UNI](https://huggingface.co/MahmoodLab/UNI). 
Due to the terms for usage of UNI, 
we are unable to directly publish the classifier's weights.
As an alternative, we provide a means of obtaining patch categories using the zero-shot classifier with [CONCH](https://github.com/mahmoodlab/CONCH).

After the patch extraction, 
users can obtain patch types by first edit the args configurations in `/data_preparation/classfication_CONCH.py`,
then run the following command
```
python /data_preparation/classfication_CONCH.py
```
Additionally, we have provided pre-prototypes calculated based on our classifier results in folder `/pre_prototypes/` 
(5 PTC classes: 0.tumor, 1.stroma, 2.infiltrate, 3.necrosis, 4.others).
According to the ablation study, 
the model still demonstrates good performance when using these pre-extracted and fixed prototypes.
If you want to use pre_prototypes for a quick start, please refer to the following step:
1. There is no need to perform the **Patch Classification** step when training the model with pre-prototypes, 
so please skip the Patch Classification step.
2. During **Heterogeneous Graph Construction** step, 
simply follow the annotation in `graph_construction.py` and set the node category path to **None** to construct the graph directly. 
3. When run `train.py`, set the "use_pre_proto" parameter to "True" in the `/configs/protosurv.yml` configuration file.

## Heterogeneous Graph Construction
We follow [Patch-GCN](https://github.com/mahmoodlab/Patch-GCN) to build the graph based on the spatial relationships among patches.
After **Patch Extraction** and **Classification** steps,
users can obtain heterogeneous graphs by first edit the path configurations in ./data_preparation/graph_construction.py, 
then run the following command.

```
python data_preparation/graph_construction.py
```

# Training ProtoSurv Model
## Make Splits
We provide the dataset lists and labels used for experiments in the /dataset_csv/ folder.
Users can split the dataset's case list into five folds for cross-validation.

```
python cross_val_split/TCGA_make_splits.py
```

## Train
The configurations yaml file is in `/configs/`. 
Users may first modify the respective config files for hyper-parameter settings, and update the path to training config in train.py,
then run the following command to start train.

```
python train.py
```



