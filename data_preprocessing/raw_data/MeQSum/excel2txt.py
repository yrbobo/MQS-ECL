import pandas as pd


def clean_text(text):
    text = text.replace('\n', '')
    text = text.replace('"', '')
    return text


excel = pd.read_excel('MeQSum.xlsx')
txt = open('MeQSum.txt', 'w', encoding='utf-8')
CHQ = excel['CHQ']
Summary = excel['Summary']
for i in range(len(CHQ)):
    CHQ[i] = clean_text(CHQ[i])
    Summary[i] = clean_text(Summary[i])
    txt.write('{}|{}\n'.format(CHQ[i], Summary[i]))

