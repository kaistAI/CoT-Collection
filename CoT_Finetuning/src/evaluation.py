from Datasets_end2end import Pretrain, check_label_name_from_json
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
import csv
import os 
import re
import string
from rouge import Rouge
from collections import Counter
import torch
import numpy as np
from collections import defaultdict
from utils import exact_match_score, accuracy_match_score_normalize, _rougel_score, ids_to_clean_text, _f1_score, remain_rationale
from tqdm import tqdm
def evaluate(args, model, tokenizer=None):
    tokenizer = model.tokenizer
    if torch.cuda.is_available() and int(args.n_gpu)>1:
        model.model.parallelize()
        device = "cuda"
        model.eval()
    elif int(args.n_gpu)==1:
        model.eval()
        device = "cuda"
        model.to('cuda')
    else:
        print("You need at least one gpu!")
        return
    
    dataloader_list = {}
    for dataset in args.eval_data:
        label_name_list = check_label_name_from_json(type_path='validation',data_dir=args.data_dir,dataset_name=dataset)
        print(label_name_list)
        for label_name in label_name_list:
            total_dataset = Pretrain(dataset=dataset, tokenizer=tokenizer, type_path='validation', input_length=args.max_input_length, 
                                    output_length=args.max_output_length, args=args, label_name=label_name)
            total_dataloader = DataLoader(total_dataset, batch_size=args.eval_batch_size, shuffle=False,num_workers=4*int(args.n_gpu), pin_memory=True)
            dataloader_list[label_name] = total_dataloader
    print('############################')
    print(dataloader_list)
    print('############################')

    if args.output_log != None:
        f = open(args.output_log, 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)

    for dataset_name,loader in tqdm(dataloader_list.items()):
        total_cnt = 0
        acc_score = 0

        for batch in iter(loader):
            with torch.no_grad():
                batch["source_ids"]=batch["source_ids"].to(device)
                batch["source_mask"]=batch["source_mask"].to(device)
                batch["target_mask"]=batch["target_mask"].to(device)
                batch["target_ids"]=batch["target_ids"].to(device)

                # select option with higer probability as answer
                if args.mode == 'rationale_evaluate':
                    if args.eval_with_prob:
                        if type(batch["label"])!=type([]):
                            output_label = batch["label"].tolist()
                        if args.peft_method == None:
                            rat = model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                use_cache=True,
                                decoder_attention_mask=batch['target_mask'].cuda(),
                                min_new_tokens=8,
                                max_new_tokens=256,
                                do_sample=True,
                                top_p=0.8,
                                no_repeat_ngram_size=3,
                                early_stopping=True
                            )
                        else:
                            rat = model.model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                use_cache=True,
                                decoder_attention_mask=batch['target_mask'].cuda(),
                                min_new_tokens=8,
                                max_new_tokens=256,
                                do_sample=True,
                                top_p=0.8,
                                no_repeat_ngram_size=3,
                                early_stopping=True
                            )
                        rat = remain_rationale(ids_to_clean_text(tokenizer, rat))

                        prob_list = []
                        with torch.no_grad():
                            for index in range(len(batch["option_list"])):
                                option = batch["option_list"]
                                verbalized_input =[r+"\n[ANSWER] "+o for r,o in zip(rat,option[index])]
                                option_ = tokenizer.batch_encode_plus(verbalized_input, max_length=args.max_output_length,
                                                                padding=True, truncation=True, return_tensors="pt")
                                lm_labels = option_["input_ids"].expand(len(batch['source_ids']), -1)
                                lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                                if args.peft_method== None:
                                    outputs = model.model(
                                        input_ids=batch["source_ids"].cuda(),
                                        attention_mask=batch["source_mask"].cuda(),
                                        labels=lm_labels.cuda(),
                                        decoder_attention_mask=option_["attention_mask"].cuda()
                                    )
                                else:
                                    outputs = model.model.model(
                                        input_ids=batch["source_ids"].cuda(),
                                        attention_mask=batch["source_mask"].cuda(),
                                        labels=lm_labels.cuda(),
                                        decoder_attention_mask=option_["attention_mask"].cuda()
                                    )
                                logits = option_["attention_mask"].cuda().unsqueeze(-1) * torch.log_softmax(outputs.logits, dim=-1)
                                lm_labels=lm_labels.cuda().unsqueeze(-1)
                                seq_token_log_prob=torch.zeros(lm_labels.shape)
                                for i in range(lm_labels.shape[0]):
                                    for j in range(lm_labels.shape[1]):
                                        seq_token_log_prob[i][j][0] = logits[i][j][lm_labels[i][j][0]]
                                seq_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
                                prob_list.append(seq_log_prob)
                            concat = torch.cat(prob_list).view(-1,len(batch['source_ids']))
                            predictions = concat.argmax(dim=0)
                            dec = [batch["option_list"][i.item()][elem_num] for elem_num, i in enumerate(predictions)] 
                        
                        # sources = ids_to_clean_text(tokenizer, batch['source_ids'])
                        targets = ids_to_clean_text(tokenizer, batch['target_ids']) 
                        # with open(f'./{args.output_log}.txt','ab') as f:
                        #     for s,t,r,d in zip(sources,targets,rat,dec):
                        #         output_string = dataset_name+"  [SEP]  "+r.replace('\n','')+"  [SEP]  "+d.replace('\n','')+"  [SEP]  "+t.replace('\n','')+"  [SEP]  "+s.replace('\n','')+"\n"
                        #         f.write(output_string.encode("utf-8"))
                    else:
                        if args.peft_method==None:
                            rat_dec = model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                use_cache=True,
                                decoder_attention_mask=batch['target_mask'].cuda(),
                                max_new_tokens=512,
                                do_sample=True,
                                top_p=0.8,
                                no_repeat_ngram_size=3,
                                early_stopping=True
                            )
                        else:
                            rat_dec = model.model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                use_cache=True,
                                decoder_attention_mask=batch['target_mask'].cuda(),
                                max_new_tokens=512,
                                do_sample=True,
                                top_p=0.8,
                                no_repeat_ngram_size=3,
                                early_stopping=True
                            )
                        dec = ids_to_clean_text(tokenizer, rat_dec)
                        dec = [gt.rsplit('[ANSWER]',1)[-1].strip() for gt in dec]
                        targets = ids_to_clean_text(tokenizer, batch['target_ids']) 
                elif args.mode=='evaluate':
                    if args.eval_with_prob:
                        if type(batch["label"])!=type([]):
                            output_label = batch["label"].tolist()
                        prob_list = []
                        for index in range(len(batch["option_list"])):
                            option = batch["option_list"]
                            option_ = tokenizer.batch_encode_plus(option[index], max_length=args.max_output_length,
                                                            padding=True, truncation=True, return_tensors="pt")
                            option_["input_ids"]=option_["input_ids"].to(device)
                            option_["attention_mask"]=option_["attention_mask"].to(device)
                            lm_labels = option_["input_ids"].expand(len(batch['source_ids']), -1)
                            lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
                            if args.peft_method==None:
                                outputs = model.model(
                                    input_ids=batch["source_ids"],
                                    attention_mask=batch["source_mask"],
                                    labels=lm_labels,
                                    decoder_attention_mask=option_["attention_mask"]
                                )
                            else:
                                outputs = model.model.model(
                                    input_ids=batch["source_ids"],
                                    attention_mask=batch["source_mask"],
                                    labels=lm_labels,
                                    decoder_attention_mask=option_["attention_mask"]
                                )
                            logits = option_["attention_mask"].unsqueeze(-1) * torch.log_softmax(outputs.logits, dim=-1)

                            lm_labels=lm_labels.unsqueeze(-1)
                            seq_token_log_prob=torch.zeros(lm_labels.shape)
                            for i in range(lm_labels.shape[0]):
                                for j in range(lm_labels.shape[1]):
                                    seq_token_log_prob[i][j][0] = logits[i][j][lm_labels[i][j][0]]
                            seq_log_prob = seq_token_log_prob.squeeze(dim=-1).sum(dim=-1)
                            prob_list.append(seq_log_prob)
                            
                        concat = torch.cat(prob_list).view(-1,len(batch['source_ids']))
                        
                        predictions = concat.argmax(dim=0)
                        dec = [batch["option_list"][i.item()][elem_num] for elem_num, i in enumerate(predictions)]
                        #sources = ids_to_clean_text(tokenizer, batch['source_ids'])
                        targets = ids_to_clean_text(tokenizer, batch['target_ids']) 
                        # with open(f'./{args.output_log}.txt','ab') as f:
                        #     for s,t,r,d in zip(sources,targets,rat,dec):
                        #         output_string = dataset_name+"  [SEP]  "+r.replace('\n','')+"  [SEP]  "+d.replace('\n','')+"  [SEP]  "+t.replace('\n','')+"  [SEP]  "+s.replace('\n','')+"\n"
                        #         f.write(output_string.encode("utf-8"))

                    else:
                        if args.peft_method==None:
                            outs = model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                use_cache=True,
                                decoder_attention_mask=batch['target_mask'].cuda(),
                                max_new_tokens=512,
                                do_sample=True,
                                top_p=0.8,
                                no_repeat_ngram_size=3,
                                early_stopping=True
                            )
                        else:
                            outs = model.model.model.generate(
                                batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                use_cache=True,
                                decoder_attention_mask=batch['target_mask'].cuda(),
                                max_new_tokens=512,
                                do_sample=True,
                                top_p=0.8,
                                no_repeat_ngram_size=3,
                                early_stopping=True
                            )
                        dec = ids_to_clean_text(tokenizer, outs)
                        targets = ids_to_clean_text(tokenizer, batch['target_ids']) 

            if (args.eval_with_prob == False): 
                total_cnt+=len(batch['source_ids'])
                for i in range(len(batch['source_ids'])):
                    ground_truth = targets[i]
                    predicted = dec[i]

                    if args.eval_with_rouge:
                        rouge = _rougel_score(predicted, ground_truth)
                        rouge_score += rouge
                        acc_score = rouge_score
                    else:
                        if ground_truth.strip().lower().replace('(','').replace(')','').replace(',','') == predicted.strip().lower().replace('(','').replace(')','').replace(',',''):
                            acc_ =1
                        else:
                            acc_ =0
                        acc_score += acc_

            else:
                total_cnt+=len(batch['source_ids'])
                acc_score += sum(list(map(lambda v: v[0] ==v[1],zip(predictions,output_label))))
                print("acc",total_cnt,acc_score)

        print(f'Name of dataset: {dataset_name}')
        print(f'Number of total validation data: {total_cnt}')
        print('Number of correct predictions: {: .0f}. Percentage : {: .4f}'.format(acc_score, acc_score/total_cnt))

        if args.eval_with_prob:
            wr.writerow([dataset_name, acc_score.item() / total_cnt,  args.checkpoint_path ])
        else:
            wr.writerow([dataset_name, acc_score / total_cnt,  args.checkpoint_path ])
    if args.output_log != None:    
        f.close()