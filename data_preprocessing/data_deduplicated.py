import json
from tqdm import tqdm


def generate_json(jsons, output):
    length = len(jsons)
    count = 1
    for j in jsons:
        json_object = '{}"chq": "{}", "faq": "{}"{}'.format('{', j['chq'], j['faq'], '}').replace('\n', '')
        if count == 1:
            output.write('[' + json_object + ',' + '\n')
        elif count == length:
            output.write(json_object + ']' + '\n')
        else:
            output.write(json_object + ',' + '\n')
        count += 1


datasets = ['MeQSum', 'iCliniq', 'HealthCareMagic']
for dataset in datasets:
    old_json = open('raw_data/{}/{}-not_deduplicated.json'.format(dataset, dataset), 'r', encoding='utf-8')
    new_json = open('raw_data/{}/{}.json'.format(dataset, dataset), 'w', encoding='utf-8')
    json_list = []
    print('对 {} 数据集进行去重'.format(dataset))
    remove_num = 0
    jsons = json.load(old_json, strict=False)
    origin_len = len(jsons)
    for j in tqdm(jsons):
        flag = True
        for k in json_list:
            if k['chq'] == j['chq']:
                flag = False
                break
        if flag:
            json_list.append(j)
        else:
            remove_num += 1
    generate_json(json_list, new_json)
    print('处理前数据量: {}, 处理后数据量: {}'.format(origin_len, origin_len - remove_num))
    old_json.close()
    new_json.close()
