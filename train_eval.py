import stanza
import torch
import time
import os
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rouge_util import Rouge_py_rouge


def train(args, model, tokenizer, hard_set, simple_set, val_set, test_set):
    train_date = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    log_dir = 'log/{}/{}'.format(args.dataset, train_date)
    log_writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir, 'train_info_{}.txt'.format(train_date)), 'w', encoding='utf-8') as f:
        f.write('model: {}, do_train_contrast: {}, contrast_decoder: {}, no_hard: {}, dataset: {}, beta: {}\n'
                .format(args.model, args.do_train_contrast, args.contrast_decoder, args.no_hard, args.dataset,
                        args.beta))
        f.close()
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.model.parameters(), lr=args.learning_rate)

    hard_len = len(hard_set)
    simple_len = len(simple_set)
    select_order = [0] * hard_len + [1] * simple_len

    best_val_rl = 0
    last_improve = 0
    for e in range(args.epochs):
        hard_set.shuffle_data()
        simple_set.shuffle_data()
        random.shuffle(select_order)
        print('Epoch [{}/{}]'.format(e + 1, args.epochs))
        sim_total = [0, 0, 0]
        sim_count = [0, 0, 0]
        total_loss = 0
        total_MLE = 0
        total_hard = 0
        total_simple = 0
        for index, i in tqdm(enumerate(select_order)):
            if i == 0:
                src_ids, tgt_ids, contrast_ids, src_mask, tgt_mask, contrast_mask, labels = hard_set.__next__()
                if args.no_hard:
                    contrast_ids = None
                    contrast_mask = None
            else:
                src_ids, tgt_ids, src_mask, tgt_mask, labels = simple_set.__next__()
                contrast_ids = None
                contrast_mask = None

            optimizer.zero_grad()
            loss, MLE_loss, loss_simple, loss_hard, sims = model(src_ids, tgt_ids, src_mask, tgt_mask, labels=labels,
                                                                 device=args.device,
                                                                 train_contrast=args.do_train_contrast,
                                                                 hard_sample_ids=contrast_ids,
                                                                 hard_sample_mask=contrast_mask, alpha=args.alpha,
                                                                 beta=args.beta,
                                                                 contrast_decoder=args.contrast_decoder,
                                                                 simcse=args.simcse, only_hard=args.only_hard)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_MLE += MLE_loss
            total_hard += loss_hard
            total_simple += loss_simple

            if len(sims) > 0:
                for idx, sim in enumerate(sims):
                    sim_total[idx] += sim.cpu().detach().item()
                    sim_count[idx] += 1

        for idx in range(len(sim_total)):
            if sim_count[idx] != 0:
                sim_total[idx] /= sim_count[idx]

        val_rg = evaluate(args, model, tokenizer, val_set)
        val_rl = val_rg[2]
        print('val set score: R1:{:.4f}, R2:{:.4f}, Rl:{:.4f}'.format(val_rg[0], val_rg[1], val_rg[2]))
        if val_rl > best_val_rl:
            best_val_rl = val_rl
            last_improve = e
            improve = '*'
            model_path = os.path.join(args.output_dir, 'best_model_{}.pt'.format(train_date))
            torch.save(model.state_dict(), model_path)
            args.model_path = model_path
            print('Best model has been updated!')
            test_score = test(args, model, tokenizer, test_set)
            log_writer.add_scalars(main_tag='test set score',
                                   tag_scalar_dict={'R1': test_score['rouge-1']['f'],
                                                    'R2': test_score['rouge-2']['f'],
                                                    'RL': test_score['rouge-l']['f']}, global_step=e)
        else:
            improve = ''
            print('No improvement in this epoch.')
            if args.save_each:
                model_path = os.path.join(args.output_dir, 'epoch_' + str(e) + '.pt')
                torch.save(model.state_dict(), model_path)

        use_time = time.time() - start_time
        log_writer.add_scalars(main_tag='loss', tag_scalar_dict={'loss': total_loss / len(select_order)}, global_step=e)
        log_writer.add_scalars(main_tag='MLE_loss', tag_scalar_dict={'MLE_loss': total_MLE / len(select_order)},
                               global_step=e)
        log_writer.add_scalars(main_tag='simple_loss',
                               tag_scalar_dict={'simple_loss': total_simple / len(select_order)},
                               global_step=e)
        if loss_hard != 0:
            log_writer.add_scalars(main_tag='hard_loss', tag_scalar_dict={'hard_loss': total_hard / len(select_order)},
                                   global_step=e)
        log_writer.add_scalars(main_tag='sim', tag_scalar_dict={'sim-pos': sim_total[0], 'sim-simple-neg': sim_total[1],
                                                                'sim-hard-neg': sim_total[2]}, global_step=e)
        log_writer.add_scalars(main_tag='val set score',
                               tag_scalar_dict={'R1': val_rg[0], 'R2': val_rg[1], 'RL': val_rg[2]}, global_step=e)
        print('epoch:{}, Train loss:{:.4f}, Sim:{}, Val rl:{:.4f}(best: {:.4f}), time:{}{}'.format(
            e + 1, total_loss / len(select_order),
            '|'.join(['{:.3f}'.format(sim) for sim in sim_total]),
            val_rl, best_val_rl, use_time, improve))

        model.train()

        if e - last_improve > args.need_improve:
            break
    log_writer.close()


def evaluate(args, model, tokenizer, val_set):
    model.eval()
    predict_all = []
    target_all = []
    with torch.no_grad():
        for src_ids, tgt_ids, src_mask, tgt_mask, labels in val_set:
            generate_ids = model.model.generate(input_ids=src_ids, attention_mask=src_mask,
                                                max_length=args.max_tgt_length, num_beams=args.num_beams,
                                                early_stopping=True)
            pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for g in generate_ids]
            tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for g in tgt_ids]
            predict_all.extend(pred)
            target_all.extend(tgt)

    score = Rouge_py_rouge(predict_all, target_all)
    scores = [score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']]
    return scores


def test(args, model, tokenizer, test_set):
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    predict_all = []
    target_all = []
    with torch.no_grad():
        for src_ids, tgt_ids, src_mask, tgt_mask, labels in test_set:
            generate_ids = model.model.generate(input_ids=src_ids, attention_mask=src_mask,
                                                max_length=args.max_tgt_length, num_beams=args.num_beams,
                                                early_stopping=True)
            pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for g in generate_ids]
            tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for g in tgt_ids]
            predict_all.extend(pred)
            target_all.extend(tgt)

    score = Rouge_py_rouge(predict_all, target_all)
    print('test set score: R1:{:.4f},R2:{:.4f},Rl:{:.4f}'.format(score['rouge-1']['f'], score['rouge-2']['f'],
                                                                 score['rouge-l']['f']))

    return score


def test_plus(args, model, tokenizer, test_set, test_set_origin):
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()
    predict_all = []
    target_all = []
    with torch.no_grad():
        for src_ids, tgt_ids, src_mask, tgt_mask, labels in test_set:
            generate_ids = model.model.generate(input_ids=src_ids, attention_mask=src_mask,
                                                max_length=args.max_tgt_length, num_beams=args.num_beams,
                                                early_stopping=True)
            pred = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for g in generate_ids]
            tgt = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for g in tgt_ids]
            predict_all.extend(pred)
            target_all.extend(tgt)

    score = Rouge_py_rouge(predict_all, target_all)
    print('test set score: R1:{:.4f},R2:{:.4f},Rl:{:.4f}'.format(score['rouge-1']['f'], score['rouge-2']['f'],
                                                                 score['rouge-l']['f']))

    with open(os.path.join(args.output_dir, 'test-predict.txt'), 'w') as f:
        for p, t, c in tqdm(zip(predict_all, target_all, test_set_origin)):
            score = Rouge_py_rouge(p, t)
            f.write('CHQ: {} | predict: {} | reference FAQ: {} | score: [R1:{}, R2:{}, RL{}]\n'
                    .format(c[0], p, t, score['rouge-1']['f'], score['rouge-2']['f'], score['rouge-l']['f']))
        f.close()

    with open(os.path.join(args.output_dir, 'test-focus.txt'), 'w') as f:
        stanza.download('en', package='mimic', processors={'ner': ['i2b2', 'BC5CDR', 'ncbi_disease', 'Radiology']})
        nlp = stanza.Pipeline('en', package='mimic',
                              processors={'ner': ['i2b2', 'BC5CDR', 'ncbi_disease', 'Radiology']})
        focus_num = 0
        total_num = 0
        for p, t in tqdm(zip(predict_all, target_all)):
            target_doc = nlp(t)
            total_num += len(target_doc.entities)
            for ent in target_doc.entities:
                if ent.text.lower() in p.lower():
                    focus_num += 1

        f.write(str(focus_num / total_num) + '\n' + str(total_num) + '\n')
        f.close()
