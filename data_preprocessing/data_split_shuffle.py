import json
import random


def generate_json(jsons, output):
    length = len(jsons)
    count = 1
    for j in jsons:
        json_object = j
        if count == 1:
            output.write('[' + json_object + ',' + '\n')
        elif count == length:
            output.write(json_object + ']' + '\n')
        else:
            output.write(json_object + ',' + '\n')
        count += 1


def process(dataset_name, train_num, val_num):
    print('开始处理数据集:{}'.format(dataset_name))
    json_data = open('raw_data/{}/{}.json'.format(dataset_name, dataset_name), encoding='utf-8')
    train = open('../dataset/{}/train.json'.format(dataset_name), 'w', encoding='utf-8')
    val = open('../dataset/{}/val.json'.format(dataset_name), 'w', encoding='utf-8')
    test = open('../dataset/{}/test.json'.format(dataset_name), 'w', encoding='utf-8')
    train_set = []
    val_set = []
    test_set = []
    dataset = json.load(json_data, strict=False)
    random.shuffle(dataset)
    idx = 1
    for d in dataset:
        json_obj = '{}"chq": "{}", "faq": "{}"{}'.format('{', d['chq'], d['faq'], '}').replace('\n', '')
        if 1 <= idx <= train_num:
            train_set.append(json_obj)
            idx += 1
        elif train_num + 1 <= idx <= train_num + val_num:
            val_set.append(json_obj)
            idx += 1
        else:
            test_set.append(json_obj)

    generate_json(train_set, train)
    generate_json(val_set, val)
    generate_json(test_set, test)
    print('{}数据集处理完毕，训练集长度: {}, 验证集长度: {}, 测试集长度: {}'.format(dataset_name, len(train_set), len(val_set), len(test_set)))


process('MeQSum', 400, 100)
process('iCliniq', 16556, 2069)
process('HealthCareMagic', 180697, 22587)

