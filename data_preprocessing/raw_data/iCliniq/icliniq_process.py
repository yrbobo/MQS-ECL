def clean_text(text):
    text = text.replace('\n', '')
    text = text.replace('"', '')
    text = text.replace('\\', '/')
    text = text.replace('•', '')
    return text


true_count = 31064
id = 0
new_file = open('iCliniq.txt', 'w', encoding='utf-8')
with open('icliniq_dialogue.txt', 'r', encoding='utf-8') as f:
    flag = False
    d_flag = False
    question = ''
    dialogue = ''
    for line in f.readlines():
        if line == 'Description\n':
            flag = True
            continue
        elif flag:
            if line[0:3] == 'Q. ':
                question += line.replace('Q. ', '')
            elif line == 'Patient:\n':
                d_flag = True
            elif d_flag:
                if line == 'Doctor:\n':
                    d_flag = False
                    flag = False
                    dialogue = clean_text(dialogue)
                    question = clean_text(question)
                    new_file.write('{}|{}'.format(dialogue, question) + '\n')
                    id += 1
                    dialogue = ''
                    question = ''
                else:
                    dialogue += line
            elif line[:3] == 'id=':
                if line[3:-1] != str(id):
                    print(line[3:])
                    print(str(id))
                    print('id not match')
    f.close()
new_file.close()
print('expect data count: {}'.format(true_count))
print('real data count: {}'.format(id))

