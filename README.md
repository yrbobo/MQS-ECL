# Medical Question Summarization with Entity-driven Contrastive Learning

This is the code repository for the paper "Medical Question Summarization with Entity-driven Contrastive Learning". The repository contains the source code and two revised datasets in the paper.

## Requirements

Python >= 3.6

pytorch == 1.10.1

transformers == 4.4.1

py-rouge == 1.1

stanza == 1.5.0

## 1. Prepare the datasets

### (1) Download the original datasets

The original datasets can be downloaded from the following URLs.

| Dataset         | URLs                                                 |
| --------------- |------------------------------------------------------|
| MeQSum          | https://github.com/abachaa/MeQSum                    |
| CHQ-Summ        | https://github.com/shwetanlp/Yahoo-CHQ-Summ          |
| iCliniq         | https://github.com/UCSD-AI4H/Medical-Dialogue-System |
| HealthCareMagic | https://github.com/UCSD-AI4H/Medical-Dialogue-System |

After the datasets are downloaded, please put each of them into a specified directory of the project.

### (2) Download the revised datasets

iCliniq and HealthCareMagic datasets suffer from a data leakage problem. For example, the duplicate rate of data samples in the iCliniq datasets reaches 33%, which leads to the overlap of training and test data, and makes the evaluation results unreliable. In order to conquer the problem, we check their data samples carefully and remove the repeated ones. The revised datasets can be downloaded from the following URLs.

| Dataset                 | URLs                                                         |
| ----------------------- | ------------------------------------------------------------ |
| iCliniq-revised         | https://drive.google.com/drive/u/1/folders/1FQTsgRYDJajcNlKJXG-FFPKFw4Cf4FzU |
| HealthCareMagic-revised | https://drive.google.com/drive/u/1/folders/1Hq4AiYr96jfOsB8OJMlyDRRUhmr_BYvY |

### (3) Remove the repeated samples from iCliniq and HealthCareMagic datasets

```python
python data_deduplicated.py
```

Note: if you have downloaded the revised datasets in last step, you can skip this one.

### (4) Preprocess datasets into a uniform formation

For different datasets, we provide the preprocessing code. You can run them to process all datasets into a uniform formation. The relevant codes are in the "data_preprocessing" directory.

## 2. Generate hard negative samples

You can generate hard negative examples for different datasets with the following command:

```python
python med_ent_perturbation.py --dataset dataset/MeQSum/train.json --output_dir dataset/MeQSum --sample_size 128 
python med_ent_perturbation.py --dataset dataset/CHQ-Summ/train.json --output_dir dataset/CHQ-Summ --sample_size 128 
python med_ent_perturbation.py --dataset dataset/iCliniq/train.json --output_dir dataset/iCliniq --sample_size 256
python med_ent_perturbation.py --dataset dataset/HealthCareMagic/train.json --output_dir dataset/HealthCareMagic --sample_size 512
```



## 3. Train

You can train the model with the following command:

```
python run.py --do_train_contrast --output_dir models/MeQSum --learning_rate 1e-5 --batch_size 16 --dataset MeQSum --epoch 20 --contrast_number 128
python run.py --do_train_contrast --output_dir models/chqsumm --learning_rate 1e-5 --batch_size 16 --dataset CHQSumm --epoch 20 --contrast_number 128
python run.py --do_train_contrast --output_dir models/iCliniq --learning_rate 1e-5 --batch_size 16 --dataset iCliniq --epoch 10 --contrast_number 128
python run.py --do_train_contrast --output_dir models/HealthCareMagic --learning_rate 1e-5 --batch_size 16 --dataset HealthCareMagic --epoch 10 --contrast_number 128
```



## 4. Test

You can test the trained model with a command like the following:

```
 python run.py --mod test --model_path models/test/model_name.pt --dataset MeQSum
```



## Acknowledgement

If this work is useful in your research, please cite our paper.

```
@article{wei2023ECL,
  title={Medical Question Summarization with Entity-driven Contrastive Learning},
  author={Sibo Wei, Wenpeng Lu, Xueping Peng, Shoujin Wang, Yi-Fei Wang, Weiyu Zhang},
  journal={arXiv preprint arXiv:2304.07437},
  year={2023}
}
```

## Checkpoints
We have uploaded the model checkpoints to HuggingFace:
https://huggingface.co/youren1999/MQS-ECL