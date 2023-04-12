import argparse
import json
import os
import secrets
import copy
import stanza
from rouge_util import Rouge_py_rouge
import sys
from tqdm import tqdm
sys.setrecursionlimit(100000)


def read_json(dataset):
    data = []
    with open(dataset, 'r', encoding='utf-8') as f:
        jsons = json.load(f, strict=False)
        for j in jsons:
            data.append(j)
        f.close()
    return data


def run(args):
    stanza.download('en', package='mimic', processors={'ner': ['i2b2', 'BC5CDR', 'ncbi_disease', 'Radiology']})
    nlp = stanza.Pipeline('en', package='mimic', processors={'ner': ['i2b2', 'BC5CDR', 'ncbi_disease', 'Radiology']})
    data = read_json(args.dataset)
    print('===construct vocabulary===')
    perturbation_list = []
    no_perturbation_list = []
    med_entities = []
    w1 = open(os.path.join(args.output_dir, 'train_perturbation.json'), 'w', encoding='utf-8')
    w2 = open(os.path.join(args.output_dir, 'train_no_perturbation.json'), 'w', encoding='utf-8')
    for obj in tqdm(data):
        faq_doc = nlp(obj['faq'])
        if len(faq_doc.entities) > 0:
            perturbation_list.append(obj)
            for ent in faq_doc.entities:
                if ent not in med_entities:
                    med_entities.append(ent)
        else:
            no_perturbation_list.append(obj)
    print('ent vocabulary length: {}'.format(len(med_entities)))
    flag = 1
    for p in tqdm(perturbation_list):
        samples = []
        faq_doc = nlp(p['faq'])
        entities = []
        for ent in faq_doc.entities:
            if ent.text not in entities:
                entities.append(ent.text)

        candidate_med_ent = list(med_entities)
        for ent in entities:
            if ent in candidate_med_ent:
                candidate_med_ent.remove(ent)

        if args.sample_size * len(entities) > len(candidate_med_ent):
            no_perturbation_list.append(p)
            continue

        rand_ent = secrets.SystemRandom().sample(candidate_med_ent, args.sample_size * len(entities))

        for i in range(args.sample_size):
            new_faq = p['faq']
            for ent in entities:
                r_ent = rand_ent.pop(0)
                samples.append(new_faq.replace(ent, r_ent.text))

        p['focus'] = entities
        p['perturbations'] = samples
        json_obj = json.dumps(p)
        if flag == 1:
            w1.write('[' + json_obj + ',' + '\n')
        elif flag == len(perturbation_list):
            w1.write(json_obj + ']' + '\n')
        else:
            w1.write(json_obj + ',' + '\n')
        flag += 1
    flag = 1
    for np in tqdm(no_perturbation_list):
        json_obj = json.dumps(np)
        if flag == 1:
            w2.write('[' + json_obj + ',' + '\n')
        elif flag == len(no_perturbation_list):
            w2.write(json_obj + ']' + '\n')
        else:
            w2.write(json_obj + ',' + '\n')
        flag += 1
    w1.close()
    w2.close()


if __name__ == '__main__':
    """
        python med_ent_perturbation.py --dataset dataset/MeQSum/train.json --output_dir dataset/MeQSum --sample_size 128 
        python med_ent_perturbation.py --dataset dataset/iCliniq/train.json --output_dir dataset/iCliniq --sample_size 256
        python med_ent_perturbation.py --dataset dataset/HealthCareMagic/train.json --output_dir dataset/HealthCareMagic --sample_size 512
    """
    parser = argparse.ArgumentParser(description="generate hard negative examples")
    parser.add_argument("--dataset", type=str, default="dataset/MeQSum/train.json", help="dataset path")
    parser.add_argument("--output_dir", type=str, default="dataset/MeQSum", help="output path")
    parser.add_argument("--sample_size", type=int, default=128, help="sample size")
    args = parser.parse_args()
    run(args)
