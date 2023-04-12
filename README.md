# 《Medical Question Summarization with Entity-driven Contrastive Learning》

## Introduction

This is the code repository for paper 《Medical Question Summarization with Entity-driven Contrastive
Learning》. The repository contains the implementation code of paper 《Medical Question Summarization with Entity-driven Contrastive
Learning》 and two newly published data sets in the paper.

## Run

### 1. Download raw datasets

The raw datasets download address of each dataset:

| dataset         | url                                                |
| --------------- | -------------------------------------------------- |
| MeQSum          | https://github.com/abachaa/MeQS                    |
| CHQ-Summ        | https://github.com/shwetanlp/Yahoo-CHQ-S           |
| iCliniq         | https://github.com/UCSD-AI4H/Medical-Dialogue-Syst |
| HealthCareMagic | https://github.com/UCSD-AI4H/Medical-Dialogue-Syst |

After the download is complete, please put the dataset in the specified directory of the project.

We also provide iCliniq and HealthCareMagic datasets that have been deduplicated, you can download them here:

| dataset             |                                                              |
| ------------------- | ------------------------------------------------------------ |
| iCliniq-new         | https://drive.google.com/drive/u/1/folders/1FQTsgRYDJajcNlKJXG-FFPKFw4Cf4FzU |
| HealthCareMagic-new | https://drive.google.com/drive/u/1/folders/1Hq4AiYr96jfOsB8OJMlyDRRUhmr_BYvY |



### 2. Requirements

Python >= 3.6

pytorch == 1.10.1

transformers == 4.4.1

py-rouge == 1.1

stanza == 1.5.0



### 3. Deduplication for iCliniq and HealthCareMagic

Please note that if you choose to directly download our processed iCliniq and HealthCareMagic datasets, you can skip this step.

```python
python data_deduplicated.py
```



### 4. Data preprocessing

For different data sets, use the data processing program we provide to process all data sets into a unified format. The relevant codes are in the 'data_preprocessing' directory.



### 5. Generate hard negative samples

You can generate hard negative examples for different datasets with the following command:

```python
python med_ent_perturbation.py --dataset dataset/MeQSum/train.json --output_dir dataset/MeQSum --sample_size 128 
python med_ent_perturbation.py --dataset dataset/CHQ-Summ/train.json --output_dir dataset/CHQ-Summ --sample_size 128 
python med_ent_perturbation.py --dataset dataset/iCliniq/train.json --output_dir dataset/iCliniq --sample_size 256
python med_ent_perturbation.py --dataset dataset/HealthCareMagic/train.json --output_dir dataset/HealthCareMagic --sample_size 512
```



### 6. Train

You can train the model with the following command:

```
python run.py --do_train_contrast --output_dir models/MeQSum --learning_rate 1e-5 --batch_size 16 --dataset MeQSum --epoch 20 --contrast_number 128
python run.py --do_train_contrast --output_dir models/chqsumm --learning_rate 1e-5 --batch_size 16 --dataset CHQSumm --epoch 20 --contrast_number 128
python run.py --do_train_contrast --output_dir models/iCliniq --learning_rate 1e-5 --batch_size 16 --dataset iCliniq --epoch 10 --contrast_number 128
python run.py --do_train_contrast --output_dir models/HealthCareMagic --learning_rate 1e-5 --batch_size 16 --dataset HealthCareMagic --epoch 10 --contrast_number 128
```



### 7. Test

You can test the trained model with a command like the following:

```
 python run.py --mod test --model_path models/test/model.pt --dataset MeQSum
```





## *We can provide a series of checkpoints for the model in the paper. If you need them, you can contact my email address: sibo.wei@foxmail.com .