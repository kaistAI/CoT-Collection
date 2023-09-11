# CoT-Collection
Repository for the paper "The CoT Collection: Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning", including 1.84M CoT rationales extracted across 1,060 tasks"

Paper Link : https://arxiv.org/abs/2305.14045

Overview of CoT Collection.
<p align="center">
  <img src="./CoT_Collection_Overview.png" width="100%" height="80%">
</p>


## Dataset Access
You could access CoT Collection via huggingface datasets library as follows:
```
from datasets import load_dataset

dataset = load_dataset("kaist-ai/CoT-Collection")
```

## Model Checkpoint Access
You could access CoT-T5 trained on CoT Collection via huggingface transformers library as follows:
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("kaist-ai/CoT-T5-11B")

model = AutoModelForSeq2SeqLM.from_pretrained("kaist-ai/CoT-T5-11B")
```

Note that we have a 11B version and a 3B version of CoT-T5!


## Code
We'll currently refactoring our code as well! We'll upload it soon:)


### Rationale Augmentation

```
sh scripts/{subset}.sh # flan, sni, t0, t0p, additional
```
Rationales for CoTCollection can be obtained by running each script corresponding to the data-subset of CoTCollection.  
  
On the first run, you will be asked to provide api keys for OpenAI.  
These keys will be saved as locally saved as api.json and be reused for future runs.  
  
Results,
- resutls for each instance during the augmentation process will be saved under  "CoT_Rationale_Augmentation/outputs/{subset}/{model_name}/rat/temp_{temperature}/"
- after all augmentation process is finished the merged data can be found at "data_extraction/data/{split}/{subset}/codex_rationale_{split}_{phase}.json"


## License
CoT Collection is only for non-commercial use and is subject to OpenAI's Terms of Use for the generated data. If you suspect any violations, please reach out to us.


## Citation
If you find this useful, please consider citing our paper:
```
@article{kim2023cot,
  title={The CoT Collection: Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning},
  author={Kim, Seungone and Joo, Se June and Kim, Doyoung and Jang, Joel and Ye, Seonghyeon and Shin, Jamin and Seo, Minjoon},
  journal={arXiv preprint arXiv:2305.14045},
  year={2023}
}
```  


## Point of contact
For any questions about the implementation or content of the paper, you could contact me via the following email:)
```
seungone@kaist.ac.kr
```
