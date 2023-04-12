def clean_text(text):
    text = text.replace('\n', '')
    text = text.replace('"', '')
    text = text.replace('\\', '/')
    text = text.replace('•', '')
    text = text.replace("|", '')
    return text


new_file = open('HealthCareMagic.txt', 'w', encoding='utf-8')
true_count = 226395
total_count = 0
for i in range(1, 5):
    print('process healthcaremagic_dialogue_{}.txt'.format(i))
    id = 0
    with open('healthcaremagic_dialogue_{}.txt'.format(i), 'r', encoding='utf-8') as f:
        flag = False
        d_flag = False
        q_flag = False
        question = ''
        dialogue = ''
        for line in f.readlines():
            if line == 'Description\n':
                q_flag = True
            elif q_flag:
                question += line
                q_flag = False
            elif line == 'Dialogue\n':
                flag = True
            elif flag:
                if line == 'Patient:\n':
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
    total_count += id
new_file.close()
print('expect data count: {}'.format(true_count))
print('real data count: {}'.format(total_count))

