import os
from tqdm import tqdm
import load_data

datasets = ['MeQSum', 'iCliniq', 'HealthCareMagic']
for dataset in datasets:
    print('check {} dataset...'.format(dataset))
    train_set, val_set, test_set = load_data.load_dataset(os.path.join('../dataset', dataset))
    val_count = 0
    test_count = 0
    for data in tqdm(val_set):
        for item in train_set:
            if data[0] == item[0]:
                val_count += 1
                print(data[0])
                print(item[0])
                break

    for data in tqdm(test_set):
        for item in train_set:
            if data[0] == item[0]:
                test_count += 1
                break
    print(val_count, test_count)
    if val_count == 0 and test_count == 0:
        print('dataset {} check √'.format(dataset))
    else:
        print('there are still problems in dataset {}'.format(dataset))