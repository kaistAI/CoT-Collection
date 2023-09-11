
import pytorch_lightning as pl
from transformers import T5Tokenizer, T5ForConditionalGeneration, MT5Tokenizer, MT5ForConditionalGeneration, Adafactor, T5Config, StoppingCriteria, StoppingCriteriaList
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, ConcatDataset
from Datasets_end2end import Pretrain, check_label_name_from_json
from utils import ids_to_clean_text, _rougel_score, calculate_accuracy_scores, remain_rationale
import torch
from torch.optim import AdamW
import os
import functools
from peft import PeftModelForSeq2SeqLM,get_peft_config
from sophia import SophiaG

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      stop_count = 0
      for stop in self.stops:
        stop_count = (stop == input_ids[0].cpu()).sum().item()

      if stop_count >= self.ENCOUNTERS:
          return True
      return False

class T5_small(pl.LightningModule):
    def __init__(self, args):
        super(T5_small, self).__init__()
        self.args = args
        if 'mt0' in args.model_name_or_path:
            self.tokenizer = MT5Tokenizer.from_pretrained(args.model_name_or_path)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        self.val_count = 0
        self.epoch = 0
        if 'mt0' in args.model_name_or_path:
            self.model = MT5ForConditionalGeneration.from_pretrained(args.model_name_or_path)    
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        if args.peft_method != None:
            print(args.peft_method)
            if self.args.peft_checkpoint_path == '':
                peft_config = {
                    "peft_type": args.peft_method,
                    "task_type": "SEQ_2_SEQ_LM",
                    "inference_mode": False,
                    "r": 4,
                    "target_modules": ["q", "v"],
                    "lora_alpha": 32,
                    "lora_dropout": 0.1,
                    "fan_in_fan_out": False,
                    "bias": "none",
                }
                peft_config = get_peft_config(peft_config)
                self.model = PeftModelForSeq2SeqLM(self.model, peft_config)
                self.model.print_trainable_parameters()
            else:
                self.model = PeftModelForSeq2SeqLM.from_pretrained(self.model, self.args.peft_checkpoint_path)

        # def set_dropout(model, dropout):
        #     """Recursively set dropout in the model."""
        #     for layer in model.children():
        #         if isinstance(layer, torch.nn.Dropout):
        #             layer.p = dropout
        #         set_dropout(layer, dropout)
        # set_dropout(self.model, 0.1)
        # print("Changing dropout rate is done!")

    def get_dataset(self, dataset, tokenizer, type_path, args):
        dataset_list = []
        if type_path == 'validation':
            label_name_list = check_label_name_from_json(type_path=type_path,data_dir=args.data_dir,dataset_name=dataset)
            for label_name in label_name_list:
                total_dataset = Pretrain(dataset=dataset, tokenizer=tokenizer, type_path=type_path, input_length=args.max_input_length, 
                                        output_length=args.max_output_length, args=args, label_name=label_name)
                dataset_list.extend(total_dataset)
        elif type_path =='train':
            dataset_list = Pretrain(dataset=dataset, tokenizer=tokenizer, type_path=type_path, input_length=args.max_input_length, 
                                        output_length=args.max_output_length, args=args, label_name=None)

        return dataset_list
        
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
    
    def _step(self, batch):

        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"]
        )

        loss = outputs[0]
        
        return loss

    def _generative_step(self, batch, batch_idx):  
        keys = batch["data_label"][0]
        if self.args.eval_with_prob == True:
            if type(batch['label']) != type(list()):
                output_label = batch["label"].tolist()
        
        accuracy_correct_num=0
        rouge_score=0
        acc_score =0   
        total_cnt = 0
    
        if self.args.mode == 'rationale_tune':

            if self.args.eval_with_prob:
                #sources = ids_to_clean_text(self.tokenizer, batch['source_ids'])
                targets = ids_to_clean_text(self.tokenizer, batch['target_ids'])
                if self.args.peft_method == None:
                    rat = self.model.generate(
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
                    rat = self.model.model.generate(
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
                rat = ids_to_clean_text(self.tokenizer, rat)
                rat = remain_rationale(rat)

                prob_list = []
                with torch.no_grad():
                    for index in range(len(batch["option_list"])):
                        option = batch["option_list"]
                        verbalized_input =[r+"\n[ANSWER] "+o for r,o in zip(rat,option[index])]
                        option_ = self.tokenizer.batch_encode_plus(verbalized_input, max_length=self.args.max_output_length,
                                                        padding=True, truncation=True, return_tensors="pt")
                        lm_labels = option_["input_ids"].expand(len(batch['source_ids']), -1)
                        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                        if self.args.peft_method == None:
                            outputs = self.model(
                                input_ids=batch["source_ids"].cuda(),
                                attention_mask=batch["source_mask"].cuda(),
                                labels=lm_labels.cuda(),
                                decoder_attention_mask=option_["attention_mask"].cuda()
                            )
                        else:
                            outputs = self.model.model(
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
                    # with open('examples.txt','a') as f:
                    #     for s,t,r,d in zip(sources,targets,rat,dec):
                    #         f.write(r+"  |  "+d+"  |  "+t+"  |  "+s+"\n")
                
            else:
                rat_dec = self.model.generate(
                    batch["source_ids"].cuda(),
                    attention_mask=batch["source_mask"].cuda(),
                    use_cache=True,
                    decoder_attention_mask=batch['target_mask'].cuda(),
                    max_new_tokens=512,
                    do_sample=True,
                    top_p=0.8,
                    #no_repeat_ngram_size=3,
                    early_stopping=True
                )
                dec = ids_to_clean_text(self.tokenizer, rat_dec)
                dec = [gt.rsplit('[ANSWER]',1)[-1].strip() for gt in dec]
                targets = ids_to_clean_text(self.tokenizer, batch['target_ids'])       

        else:
            if self.args.eval_with_prob:
                #sources = ids_to_clean_text(self.tokenizer, batch['source_ids'])
                targets = ids_to_clean_text(self.tokenizer, batch['target_ids']) 
                prob_list = []
                with torch.no_grad():
                    for index in range(len(batch["option_list"])):
                        option = batch["option_list"]
                        option_ = self.tokenizer.batch_encode_plus(option[index], max_length=self.args.max_output_length,
                                                        padding=True, truncation=True, return_tensors="pt")
                        lm_labels = option_["input_ids"].expand(len(batch['source_ids']), -1)
                        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
                        outputs = self.model(
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

                    # with open('examples.txt','a') as f:
                    #     for s,t,d in zip(sources,targets,dec):
                    #         f.write(d+"  |  "+t+"  |  "+s+"\n")
            else:
                outs = self.model.generate(
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
                dec = ids_to_clean_text(self.tokenizer, outs)
            #texts = [self.tokenizer.decode(ids) for ids in batch['source_ids']]
            targets = ids_to_clean_text(self.tokenizer, batch['target_ids']) 

        if (self.args.eval_with_prob == False): 
            for i in range(len(batch['source_ids'])):
                total_cnt+=1
                ground_truth = targets[i]
                predicted = dec[i]
                if self.args.eval_with_rouge:
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
            acc_score = sum(list(map(lambda v: v[0] ==v[1],zip(predictions,output_label))))
                
        return acc_score, total_cnt


    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = Adafactor(optimizer_grouped_parameters, lr=self.args.learning_rate, scale_parameter=False, relative_step=False)
        #optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate,weight_decay=0.0)
        #Reset the optimizer state
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state[p]
                if 'step' in state:
                    state['step'] = 0
        

        if self.args.use_lr_scheduling:
            len_data = len(self.train_dataloader())
            denomniator = (self.args.n_gpu * self.args.gradient_accumulation_steps)
            steps_per_epoch = ( len_data // denomniator ) + 1
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.005, epochs=self.args.num_train_epochs, anneal_strategy='linear', cycle_momentum=False)
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "name": "learning rate"}]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss = self._step(batch)
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.model.eval()
        accs_dict = {}
        accuracy, cnt = self._generative_step(batch, batch_idx)
        return {batch['data_label'][0]: [accuracy, cnt]}
    def test_step(self, batch, batch_idx):
        for keys in batch.keys():
            self._generative_step(batch, keys, batch_idx)



    def validation_epoch_end(self, validation_step_outputs):
        
        score_dict = {}
        score = 0
        
        validation_step_outputs_gather = self.all_gather(validation_step_outputs)
        if type(self.args.eval_data)==list and len(self.args.eval_data)>1:
            for datas in validation_step_outputs_gather:
                for output in datas:
                    for key, [accs, cnt] in output.items():
                        if key not in score_dict.keys():
                            score_dict[key]=[accs, cnt]
                        else:
                            old_acc, old_cnt = score_dict[key]
                            score_dict[key] = [old_acc + accs, old_cnt + cnt]
        else:
            for output in validation_step_outputs_gather:
                for key, [accs, cnt] in output.items():
                    if key not in score_dict.keys():
                        score_dict[key]=[accs, cnt]
                    else:
                        old_acc, old_cnt = score_dict[key]
                        score_dict[key] = [old_acc + accs, old_cnt + cnt]
        for key, [accs, cnt] in score_dict.items():
            self.log(f'acc_score_{key}', (accs/cnt).mean(), prog_bar=True, logger=True, sync_dist=True)
            score += accs/cnt
        
        score = score / len(score_dict)

        self.agg_score = score
        print("agg_score is", self.agg_score.mean(), self.val_count)
        self.log(f'acc_score_mean', self.agg_score.mean(), prog_bar=True, logger=True, sync_dist=True)
        # if self.val_count > 3:
            # print("enter")
        if self.args.peft_method == None:
            param_dict = {}
            for name, param in self.model.named_parameters():
                param_dict[name]=param.clone().detach().cpu()
            torch.save(param_dict, self.args.output_dir[:-3]+'_'+self.args.wandb_run_name+'.pt') 


    def train_dataloader(self):
        total_dataset = []
        for dataset in self.args.train_data:
            dataset_elem = self.get_dataset(dataset=dataset, tokenizer=self.tokenizer, type_path="train", args=self.args)
            total_dataset.append(dataset_elem)
        train_dataset = ConcatDataset(total_dataset)
        print('$'*50)
        print(train_dataset)
        print('$'*50)

        
        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(train_dataset, sampler=sampler,  batch_size=self.args.train_batch_size, drop_last=False, num_workers=self.args.num_workers)
        return dataloader

    def val_dataloader(self):

        dataloader_list = []

        for dataset in self.args.eval_data:
            dataset_elem = self.get_dataset(dataset=dataset, tokenizer=self.tokenizer, type_path="validation", args=self.args)
            dataloader_list.append(DataLoader(dataset_elem, batch_size=self.args.eval_batch_size, drop_last=False, num_workers=self.args.num_workers))

        return dataloader_list

    def test_dataloader(self):
        return self.val_dataloader()