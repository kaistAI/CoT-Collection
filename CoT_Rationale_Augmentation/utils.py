import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import sleep

import openai
import tiktoken
from tqdm import tqdm


import json
import os 
from argparse import ArgumentParser
from collections import defaultdict,OrderedDict
from prettytable import PrettyTable
import copy

class LLM:
    def __init__(self, args, api_key, aug_data):
        self.args = args
        self.login_to_openai(api_key)  # Login with the first key
        # self.data_int = AugmentedDataset(args)
        self.data_int = aug_data
        self.encoding = tiktoken.encoding_for_model(args.model_name)

    def login_to_openai(self, key):
        openai.api_key = key

    def file_exists(self, directory: Path, filename: str) -> bool:
        return (directory / filename).exists()
    
    def make_dir(self, dir_path):
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    

    def configure_savepath(self, category: str, task: str, prompt_name: str) :

        coverage_dir = Path(self.args.base_dir) / 'outputs' / self.args.data_coverage / self.args.model_name
        rat_dir = self.make_dir(coverage_dir / 'rat' / f'temp_{self.args.temperature}' / category / task / prompt_name)
        raw_dir = self.make_dir(coverage_dir / 'raw' / f'temp_{self.args.temperature}' / category / task / prompt_name)
        return rat_dir, raw_dir
    
    
    def _handle_context_length_error(self, input):
        num_demonstrations = input.count("[Example") - 1 
        remove_dem_start = input.find(f"[Example {num_demonstrations}]")
        remove_dem_end = input.find(f"[Example {num_demonstrations + 1}]")
        input = input[:remove_dem_start] + input[remove_dem_end:]
        input = input.replace(f"[Example {num_demonstrations+1}]", f"[Example {num_demonstrations}]")
        return input
    
    def _handle_timeout_error(self, timeout_stack):
        timeout_stack += 1
        if timeout_stack > 10:
            timeout_stack = 1
        time.sleep(timeout_stack)
        return timeout_stack

        
    def response_parser(self, outputs):
        results = []

        for output in outputs:
            try:
                explanation = output.strip()
            except:
                explanation = ""
            results.append(
                explanation
            )
        return results

    def _openai_completion(self, input):
        response = openai.Completion.create(
            engine=self.args.model_name,
            prompt=input,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.num_samples,
            presence_penalty=self.args.presence_penalty,
            frequency_penalty=self.args.frequency_penalty,
            max_tokens=self.args.max_tokens,
            stop=["\n\n\n\n","\n\n\n"],
            logprobs=1
        )
        return response

    def _openai_chat(self, input):
        response = openai.ChatCompletion.create(
            model=self.args.model_name,
            messages = [{"role": "user", "content": f"{input}"}],
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            n=self.args.num_samples,
            presence_penalty=self.args.presence_penalty,
            frequency_penalty=self.args.frequency_penalty,
            max_tokens=self.args.max_tokens,
            stop=["\n\n\n\n","\n\n\n",'[DONE]'],
        )
        return response
    

    def inference(self, input):
        timeout_stack = 0
        while True:
            try:
                if self.args.model_name == "code-davinci-002":
                    response = self._openai_completion(input)
                else:
                    response = self._openai_chat(input)
                break
            except Exception as e:
                print(e)
                if "maximum context length" in str(e):
                    input = self._handle_context_length_error(input)
                else:
                    timeout_stack = self._handle_timeout_error(timeout_stack)
        return response


    def get_rationale(self, prompt_input):
        _llm_inference = self.inference(prompt_input)
        
        llm_inference = _llm_inference['choices'] # type: ignore

        if self.args.model_name == "code-davinci-002":
            outputs = [o['text'] for o in llm_inference]
        else:
            outputs = [o['message']['content'] for o in llm_inference]

        results = self.response_parser(outputs)


        return results, _llm_inference


    def gen_and_save(self, _data, input, idx, RAT_DIR, RAW_DIR):
        rationale, raw = self.get_rationale(input)
        _data['rationale'] = rationale

        with open(RAT_DIR / f"{idx}.json", "w") as f:
            json.dump(_data, f, indent=4)

        with open(RAW_DIR / f"{idx}.json", "w") as f:
            json.dump(raw, f, indent=4)



    def run(self):
        if self.args.mode == 'multi':
            with ProcessPoolExecutor(max_workers=self.args.request_per_minute) as executor:
                requests_per_minute_total = self.args.request_per_minute
                for (inst,input) in tqdm(self.data_int):
                    idx = inst['idx']

                    RAT_DIR, RAW_DIR = self.configure_savepath(inst['category'],inst['task'],inst['prompt_name'])
                    if self.file_exists(RAT_DIR, f"{idx}.json"):
                        print("exists", RAT_DIR, idx)
                        continue
                    else:
                        tokens = self.encoding.encode(input)
                        if len(tokens) > 3900:
                            print("too long",RAT_DIR, idx)
                            continue
                        executor.submit(self.gen_and_save,inst['data'], input, idx, RAT_DIR, RAW_DIR)
                        sleep(1 / requests_per_minute_total * 60)

        elif self.args.mode == 'single':
            for (inst,input) in tqdm(self.data_int):
                idx = inst['idx']


                RAT_DIR, RAW_DIR = self.configure_savepath(inst['category'],inst['task'],inst['prompt_name'])
                if self.file_exists(RAT_DIR, f"{idx}.json"):
                    print("exists", RAT_DIR, idx)
                    continue
                else:
                    tokens = self.encoding.encode(input)
                    if len(tokens) > 3900:
                        print("too long",RAT_DIR, idx)
                        continue
                    self.gen_and_save(inst['data'], input, idx, RAT_DIR, RAW_DIR)




class AugmentedDataset:
    def __init__(self,args, idx: int, num_keys: int):
        self.args = args
        self.data = self.load_dataset()
        self.idx = 0
        if num_keys == 1:
            print('using entire dataset')
        else:
            print(f'using partial dataset {idx} of {num_keys}')
            self.data = self.data[idx::num_keys]
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx < self.__len__():
            inst = self.data[self.idx]
            self.idx += 1
            input = self.prepare_input(inst['category'], self.load_prompt(inst['category'], inst['task']), inst['data'])
            return inst, input 
        else:
            raise StopIteration

    def load_dataset(self, ):
        instance = []
        for data_file in self.args.data_file.split('+'):
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            if self.args.data_coverage in 'few_shot':
                ## for raft
                phase = data_file.split('/')[-1].split('.')[0]
                category = phase
            else:
                phase = data_file.split('/')[-1].split('_')[0]
                category = data_file.split('/')[-1].split('_')[1]
            
            for task,v in data.items():
                for prompt_name, _v in v.items():
                    for idx, data in _v.items():
                        instance.append({'category':category,'task':task,'prompt_name':prompt_name,'idx':idx,'data':data})
        return instance 

    def load_prompt(self, category, task):
        if self.args.data_coverage == 'translation':
            prompt = f"You're the best translator in the world. Given the following question and answer, translate English into {self.args.split}.\n\n"
        elif self.args.data_coverage == 'few_shot':
            with open(f'./demonstration/few_shot/{category}.txt','r') as f:
                prompt = f.read()
        elif self.args.data_coverage =='sni':
            with open(f"./demonstration/sni/{category}.txt", 'r') as f:
                prompt = f.read()
        elif category == 'misc':
            with open(f"./demonstration/misc/{task}.txt", 'r') as f:
                prompt = f.read()
        elif task == 'drop':
            with open(f"./demonstration/rc/drop.txt", 'r') as f:
                prompt = f.read()
        elif task == 'aqua':
            with open(f"./demonstration/math/aqua.txt", 'r') as f:
                prompt = f.read()
        elif 'nli' in task:
            with open(f"./demonstration/nli/prompt.txt", 'r') as f:
                prompt = f.read()
        elif 'qa' in task:
            with open(f"./demonstration/exqa/prompt.txt", 'r') as f:
                prompt = f.read()
        else:
            try:
                with open(f"./demonstration/{category}/prompt.txt", 'r') as f:
                    prompt = f.read()
            except:
                with open(f"./demonstration/mcqa/prompt.txt", 'r') as f:
                    prompt = f.read()
        return prompt
    
    def prepare_input(self, category, demonstration, data):
        if category == "mcqa":
            if 'labels_list' in data.keys():
                labels = ""
                for label in data['labels_list']:
                    labels += f"- {label}\n"
            else: 
                labels = "Not Given\n"
            input = demonstration + str(data['source']).strip()+ "\n\n[Options]\n"+ labels+"\n\n[Answer]\n" + str(data['target']).strip() +"\n\n[Rationale]\n"
        elif self.args.data_coverage == 'few_shot':
            options = '\n-'+'\n-'.join(data['labels_list'])
            input = demonstration + f'''
Input: {data['source']}
Possible Options: {options}
Output: {data['target']}
Annotated Rationale:  '''
        elif self.args.data_coverage == 'translation':
            input = demonstration + f"[Question]\n{data['source'].strip()}\n\n[Answer]\n{data['target'].strip()}"
        else: 
            input = demonstration + str(data['source']).strip() +"\n\n[Answer]\n" + str(data['target']).strip() +"\n\n[Rationale]\n"
        return input
    
    def split_data(self,idx,num_keys):
        return AugmentedDataset(self.args, idx, num_keys)
    
    def configure_savepath(self,) :
        save_cats = set()
        for i in self.data:
            coverage_dir = Path(self.args.base_dir) / 'outputs' / self.args.data_coverage / self.args.model_name
            rat_dir = coverage_dir / 'rat' / f'temp_{self.args.temperature}' / i['category'] 
            save_cats.add(rat_dir)
        merge_dir = Path(self.args.base_dir) / 'merged_outputs' / self.args.data_coverage
        merge_dir.mkdir(parents=True, exist_ok=True)

        return list(save_cats), merge_dir
    
    

    def walk_dir(self):
        result = {}
        saved_dirs, new_dir = self.configure_savepath()
        for base_dir in saved_dirs:
            dir_tree = defaultdict(dict)
            for cat in sorted(os.listdir(base_dir)):
                for prompt in sorted(os.listdir(os.path.join(base_dir,cat))):
                    dir_tree[str(cat)][str(prompt)] = {} # type: ignore
                    if self.args.data_coverage == "translate":
                        int_idxs = [i.split('.')[0] for i in os.listdir(os.path.join(base_dir,cat,prompt))]
                        idxs = [str(i) for i in int_idxs]
                    else:
                        int_idxs = sorted([int(i.split('.')[0]) for i in os.listdir(os.path.join(base_dir,cat,prompt))])
                        idxs = [str(i) for i in int_idxs] 

                    for file in idxs:
                        try:
                            with open(os.path.join(base_dir,cat,prompt,f"{file}.json"), "r") as f:
                                data = json.load(f)

                                if self.args.data_coverage == "translate":
                                    if data['rationale'][0] == '':
                                        continue
                                    else:
                                        data['source'] = "".join(data['rationale'][0]['question'].split('[')[:-1]).strip()
                                        data['target'] = "".join(data['rationale'][0]['answer'].split('[')[0]).strip()
                                        data['rationale'] = [data['rationale'][0]['rationale'].strip()]
                                else:
                                    for _filt in FILTER:
                                        filtered_rationale = []
                                        for rat in data['rationale']:
                                            if rat == '':
                                                filtered_rationale.append(rat)
                                            else:
                                                filtered_rationale.append(rat.split(_filt)[0])
                                        data['rationale'] = filtered_rationale
                                if len(data['rationale']) == 0:
                                    continue
                                else:
                                    rats = copy.deepcopy(data['rationale'])
                                    for idx, rat in enumerate(rats):
                                        data['rationale'] = rat
                                        dir_tree[str(cat)][str(prompt)][str(file)+"_"+str(idx)] = data # type: ignore
                        except Exception as e:
                            print(e)
                            print(os.path.join(base_dir,cat,prompt,f"{file}.json"))
                            break
            result[str(base_dir)] = dir_tree
        return result, new_dir

    def filter_instance(self,rat):
        for _filt in INSTANCE_FILTER:
            if _filt not in rat:
                return ''
        if rat[0].isalpha():
            if len(rat) > 25:
                return rat
        return ''


    def merge_results(self):
        total, new_dir = self.walk_dir()
        for k,v in total.items():
            category = k.split("/")[-1].split("_")[0]
            print(k)
            filename = f"{self.args.split}.json"
            
            with open(new_dir / f"{self.args.data_coverage}_rationale_{category}_{filename}", "w") as f:
                json.dump(v, f, indent=4)
        return "DONE"

    def filter(self, data):
        def filter1(data):
            result = []
            target = data['target']
            for rat in data['rationale']: ## answer in the first sentence of the rationale
                if target in rat.split('.')[0]:
                    result.append(0)
                else:
                    result.append(1)
            return result

        def filter2(data):
            result = []
            target = data['target']
            for rat in data['rationale']:
                if target not in rat: ## answer isn't within rationale
                    result.append(0)
                else:
                    result.append(1)
            return result

        def filter3(data):
            punc =['.',',','!','?']
            result = []
            for rat in data['rationale']:
                try:
                    if rat.strip()[-1] not in punc: ## rationale doesn't end as a sentence
                        result.append(0)
                    else:
                        result.append(1)
                except:
                    # print(data)
                    result.append(0)
            return result

        def filter5(data):
            result = []
            for rat in data['rationale']:
                num = sum(rat==i for i in data['rationale'])
                if num >= 2: ## rationale is repeated
                    result.append(0)
                else:
                    result.append(1)
            return result
        
        f1 = filter1(data)
        f2 = filter2(data)
        f3 = filter3(data)
        f5 = filter5(data)
        all = [_f1*_f2*_f3*_f5 for _f1,_f2,_f3,_f5 in zip(f1,f2,f3,f5)]


        return {'f1':f1,'f2':f2,'f3':f3,'f5':f5,
                'all':all}





FILTER = [
    "\n`\n\n'''",
    "\n`\n",
    "\n''\n",
    "\n'''\n",
    "\n```",
    "\n\n \n'",
    "\n\n \n`",
    "\n\n \nimport",
    "\n`;\n\n",
    "\"\n\n",
    "[examp",
    "[Examp",
    "\n`;\n\n",
    "'''\n",
    "\n`\n",
    "\n\n``",
    "\n``",
    "\"\"\"\n",
    "\n\n\t",
    "\n#",
    "\";\n",
    "\"\n\t",
    "print(",
    "\n          ",
    "\"\n'''",
    "'''\nimport",
    "\"\n\n\t",
    "\n\n      ",
    "\n\t\t",
    "\t        ",
    "\"\n    }\n",
    "\n\n  ####",
    "\n\n \t`)\n}",
    "\n</block>\n\n",
    "\n\n         */\n",
    "\"\n\n \n  \t`;",
    "\n\n \t",
    "\"\n\n \t */",
    "\";\n\n    }",
    "\n\n    \t\t",
    "\"\n`,\n\t}",
    "]]\n\t",
    "\"\n\n`",
    "\"\n'''\n\n",
    "\n\n             OR",
    "\n \n"
]
INSTANCE_FILTER = [
    "[Answer]",
    "[Rationale]",
]