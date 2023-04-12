dataset = 'MeQSum'
txt_name = 'raw_data/{}/{}.txt'.format(dataset, dataset)
json_name = 'raw_data/{}/{}-not_deduplicated.json'.format(dataset, dataset)
json_file = open(json_name, 'w', encoding='utf-8')


def clean_text(text):
    text = text.replace('\n', '')
    text = text.replace('"', '')
    text = text.replace('\\', '/')
    return text


with open(txt_name, encoding='utf-8') as f:
    lines = f.readlines()
    length = len(lines)
    count = 1
    for line in lines:
        if len(line.split('|')) > 1:
            chq, faq = line.split('|')
            chq = clean_text(chq)
            faq = clean_text(faq)
        else:
            print('data error: {}'.format(line))
            break
        json_object = '{}"chq": "{}", "faq": "{}"{}'.format('{', chq, faq, '}')
        if count == 1:
            json_file.write('[' + json_object + ',' + '\n')
            count += 1
        elif count == length:
            json_file.write(json_object + ']' + '\n')
        else:
            json_file.write(json_object + ',' + '\n')
            count += 1
    f.close()
json_file.close()



