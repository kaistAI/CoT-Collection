from torch.utils.data import Dataset
import random
import torch
import json
import datasets
from datasets import load_dataset, DatasetDict
import os



def label_(options, answer):
    try:
        label = options.index(answer)
    except:
        print('@@@@@@@@')
        print(options)
        print()
        print(answer)
        print('@@@@@@@@')
    return label

def check_label_name_from_json(type_path, data_dir, dataset_name):
    label_name_list = []
    _file = open(os.path.join(data_dir,type_path,dataset_name+'.json'))
    _data = json.load(_file)
    if ('t0' in dataset_name) or ('train' in dataset_name):
        for dn in _data:
            for pn in _data[dn]:
                label_name_list.append(dn+"@"+pn)
    else:
        for dn in _data:
            label_name_list.append(dn)
    return label_name_list

def load_validation_dataset_from_json(type_path, data_dir, dataset_name, label_name):
    _file = open(os.path.join(data_dir,type_path,dataset_name+'.json'))
    _data = json.load(_file)
    instances = []
    if ('t0' in dataset_name) or ('train' in dataset_name):
        for dn in _data:
            for pn in _data[dn]:
                if dn+"@"+pn == label_name:
                    for idx in _data[dn][pn]:
                        instances.append(_data[dn][pn][idx])
    else:
        for dn in _data:
            if dn == label_name:
                for idx in _data[dn]:
                    instances.append(_data[dn][idx])
    random.shuffle(instances)
    
    return datasets.Dataset.from_list(instances)

def load_train_dataset_from_json(type_path, data_dir, dataset_name):
    _file = open(os.path.join(data_dir,type_path,dataset_name+'.json'))
    _data = json.load(_file)
    instances = []
    if ('collection' in dataset_name) or ('casehold' in dataset_name) or ('case_hold' in dataset_name) or ('ledgar' in dataset_name) or ('mednli' in dataset_name) or ('pubmedqa' in dataset_name) or ('pubmed_qa' in dataset_name):
        for idx in _data:
            instances.append(_data[idx])
    elif ('t0' in dataset_name) or ('train' in dataset_name) or ('raft' in dataset_name):
        for dn in _data:
            for pn in _data[dn]:
                for idx in _data[dn][pn]:
                    instances.append(_data[dn][pn][idx])
    else:
        for dn in _data:
            for idx in _data[dn]:
                instances.append(_data[dn][idx])
    random.shuffle(instances)
    
    return datasets.Dataset.from_list(instances)

class Pretrain(Dataset):
    def __init__(self, dataset, tokenizer, type_path, input_length, output_length, args, label_name):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path
        # self.validation_per = args.v
        
        self.data_dir = args.data_dir
        if label_name == None:
            self.whole_dataset = dataset
        else:
            self.whole_dataset = dataset+"@"+label_name

        if self.type_path == 'train':
            self.dataset = load_train_dataset_from_json(type_path,self.data_dir,dataset)
        elif self.type_path == 'validation':
            self.dataset = load_validation_dataset_from_json(type_path,self.data_dir,dataset,label_name)
        
        print(f'Length of {self.whole_dataset} retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length
        #self.ids_to_answers = ids_to_answers

    def __len__(self):
        return len(self.dataset)


    def convert_to_feature_tokenizer(self, input_, target_, options):
        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                            padding='max_length', truncation=True, return_tensors="pt", add_special_tokens=True)

        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")

        data_label = self.whole_dataset 
        return source, targets, data_label, options
        

    def convert_to_features_binary(self, example_batch, index):
        source_ids_batch=[]
        target_ids_batch=[]
        src_mask_batch=[]
        target_mask_batch=[]

        input_ = example_batch['source'].strip()
        answer = example_batch['target'].strip()
        if 'labels_list' in example_batch:
            options = example_batch['labels_list']

        good_label = label_(options, answer)
        bad_label = random.choice([i for i in range(0,len(options)) if i!=good_label])
        # print("option_idx, label", option_idx, label)
        gold_label = options[good_label]
        black_label = options[bad_label]
        
        target_ = gold_label

        source, targets, data_label, options= self.convert_to_feature_tokenizer(input_, target_, options)
        target_ids = targets["input_ids"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        
        target_ids_batch.append(target_ids)
        target_mask_batch.append(target_mask)

        
        target_ = black_label
        source, targets, data_label, options= self.convert_to_feature_tokenizer(input_, target_, options)
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()
        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        target_ids_batch.append(target_ids)
        target_mask_batch.append(target_mask)
        source_ids_batch = source_ids 
        src_mask_batch = src_mask
        return source_ids_batch ,src_mask_batch, target_ids_batch, target_mask_batch, data_label, options, good_label


    def convert_to_features(self, example_batch, index):
        input_ = example_batch['source'].strip()
        target_ = example_batch['target']
        if 'labels_list' in example_batch:
            options = example_batch['labels_list']
            if options != None:
                try:
                    label = label_(options, target_)
                except:
                    label =target_
            else:
                label = target_
        else:
            options = None
            label = target_

        target_ = target_
        if self.args.mode == 'rationale_tune':
            if self.type_path =='validation':
                prefix = "\n\nLet's think step by step.\n"
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

                # # Calculate the maximum allowed length for input tokens
                max_input_length = self.input_length - len(prefix_tokens)

                input_tokens = self.tokenizer.encode(input_, max_length=max_input_length, truncation=True)
                input_ = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

                input_ = input_ + prefix
                
            else:
                if ('rationale' in example_batch):
                    if example_batch['rationale'] is not None:
                        prefix = "\n\nLet's think step by step.\n"
                        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

                        # # Calculate the maximum allowed length for input tokens
                        max_input_length = self.input_length - len(prefix_tokens)

                        input_tokens = self.tokenizer.encode(input_, max_length=max_input_length, truncation=True)
                        input_ = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

                        input_ = input_ + prefix
                        
                        explanation_ = example_batch['rationale']
                        if explanation_[-1] != ".":
                            explanation_ += "."
                        explanation_prefix = "\n[ANSWER] "
                        
                        # Tokenize the explanation prefix and target without special tokens
                        explanation_prefix_tokens = self.tokenizer.encode(explanation_prefix, add_special_tokens=False)
                        target_tokens = self.tokenizer.encode(target_, add_special_tokens=False)

                        # Calculate the maximum allowed length for explanation tokens
                        max_explanation_length = self.output_length - len(explanation_prefix_tokens) - len(target_tokens)

                        explanation_tokens = self.tokenizer.encode(explanation_, max_length=max(max_explanation_length,0), truncation=True)
                        explanation_ = self.tokenizer.decode(explanation_tokens, skip_special_tokens=True)

                        target_ = explanation_ + explanation_prefix + target_
                    else:
                        prefix = "\n[ANSWER] "
                        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

                        # Calculate the maximum allowed length for input tokens
                        max_input_length = self.input_length - len(prefix_tokens)

                        input_tokens = self.tokenizer.encode(input_, max_length=max_input_length, truncation=True)
                        input_ = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

                        input_ = input_ + prefix
                else:
                    prefix = "\n[ANSWER] "
                    prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

                    #Calculate the maximum allowed length for input tokens
                    max_input_length = self.input_length - len(prefix_tokens)

                    input_tokens = self.tokenizer.encode(input_, max_length=max_input_length, truncation=True)
                    input_ = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

                    input_ = input_ + prefix
        elif self.args.mode == 'finetune':
            if ('rationale' in example_batch):
                prefix = "\n\nLet's think step by step.\n"
                input_ = input_ + prefix
                
                explanation_ = example_batch['rationale'].strip()
                explanation_prefix = "\n[ANSWER] "

                target_ = explanation_ + explanation_prefix + target_
            else:
                prefix = "\n[ANSWER] "

                input_ = input_ + prefix
        elif (self.args.mode == 'evaluate') or (self.args.mode == 'decoder_evaluate'):
            prefix = "\n[ANSWER] "
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

            #Calculate the maximum allowed length for input tokens
            max_input_length = self.input_length - len(prefix_tokens)

            input_tokens = self.tokenizer.encode(input_, max_length=max_input_length, truncation=True)
            input_ = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

            input_ = input_ + prefix
        elif (self.args.mode == 'rationale_evaluate') or self.args.mode == 'decoder_rationale_evaluate':
            prefix = "\n\nLet's think step by step.\n"
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

            #Calculate the maximum allowed length for input tokens
            max_input_length = self.input_length - len(prefix_tokens)

            input_tokens = self.tokenizer.encode(input_, max_length=max_input_length, truncation=True)
            input_ = self.tokenizer.decode(input_tokens, skip_special_tokens=True)

            input_ = input_ + prefix

        source, targets, data_label, options = self.convert_to_feature_tokenizer(input_, target_, options)
        return source, targets, data_label, options, label

    def __getitem__(self, index):
        indexed_data = self.dataset[index]    

        source, targets, data_label, options, label = self.convert_to_features(indexed_data, index)
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()
        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()
        if options is not None:
            option_list = options
        else:
            option_list = -1
        if (self.type_path =='validation') and (self.args.eval_with_prob):
            #print(f"{self.type_path} Dataset with Option List loaded")
            return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "data_label": data_label, "option_list": option_list, "label": label, "index":str(int(index)+50)}
        else:
            #print(f"{self.type_path} Dataset without Option List loaded")
            return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "data_label": data_label}
        
        
