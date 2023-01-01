# GEC-BART
This is an unofficial implementation of the following paper:

```
@inproceedings{katsumata-komachi-2020-stronger,
    title = "Stronger Baselines for Grammatical Error Correction Using a Pretrained Encoder-Decoder Model",
    author = "Katsumata, Satoru  and
      Komachi, Mamoru",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    month = dec,
    year = "2020",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.aacl-main.83",
    pages = "827--832",
}
```

### Install

Confirmed that it works on python==3.8.10 and following modules:
```bash
pip install transformers==4.23.1 accelerate==0.13.2
```

### Training

This implementation uses Accelerate module for training on multiple GPUs.  
Please execute `accelerate config` and input settings.  
Here is an example for training on a GPU.
```bash
accelerate config
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
# Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 0
# Do you want to run your training on CPU only (even if a GPU is available)? [yes/NO]:
# Do you want to use DeepSpeed? [yes/NO]: 
# What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
# Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: bf16
```

Here is a quick start example. You can use your data by changing `source`, `target` and `{train|valid}pref` option.
```bash
CUDA_VISIBLE_DEVICES=0 \
accelerate launch train.py \
--model_id facebook/bart-base \
--source src \
--target trg \
--trainpref demo/train \
--validpref demo/train \
--outdir models/sample \
--seed 1 \
--epoch 5 \
--batch_size 1 \
--accumulation 1
```

The training progress will be shown as the following format:
```
[Epoch 0] [TRAIN]:  12%|█▏        | 520/4387 [02:31<20:56,  3.08it/s, loss=2.56, lr=7.8e-6]
```

The trained models are saved the following format.  
`best/` is the checkpoint that achieves lowest loss on the validation data.
```
model/
├── best
│   ├── config.json
│   ├── lr_state.bin
│   ├── merges.txt
│   ├── my_config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.json
├── last
│   ├── ...
└── log.json
```

### Predict / Generate

```
python generate.py \
 --restore_dir <best/ or last/ dir> \
 --input <a raw text file that you want to correct errors> \
 > output_file.txt
```

### Note: Label smoothing of CrossEntropyLoss

A Label smoothing is important but Huggingface's `BartForConditionGeneration` does not support the label smoothing option.

So, it's better to directly change `loss_fct = CrossEntropyLoss()` to `loss_fct = CrossEntropyLoss(label_smoothing=0.1)` in `transformers/models/bart/modeling_bart.py`. This slightly improves performances.

### Performances obtained

I conducted an experiment using this implementation. The parameters are:
|hyperparameter|value|
|:--|:--|
|Pretrained Model|`facebook/bart-base` or `facebook/bart-large`|
|Train data|BEA19-train without non-corrected pairs (about 560k pairs)|
|Validation data|FCE-dev|
|Optimizer|Adam|
|Learning rate|3e-5|
|Learning rate scheduler|linear|
|Warmup steps for lr-scheculer|500|  
|Batch size|64|
|Gradients accumulation|4|
|Max length (Both Enc and Dec)|128|
|Label smoothing of CrossEntropyLoss|0.1|
|Num epochs|5|

The trained models are available from Huggingface Hub:  
`gotutiyan/gec-bart-base`: [model card](https://huggingface.co/gotutiyan/gec-bart-base)  
`gotutiyan/gec-bart-large`: [model card](https://huggingface.co/gotutiyan/gec-bart-large)

The performances of the models are:
|Data|Metric|gotutiyan/gec-bart-base|gotutiyan/gec-bart-large|Paper (bart-large)|
|:--|:--|:--|:--|:--|
|CoNLL-2014|M2 (P/R/F0.5)|70.0 / 38.5 / 60.2|71.01 / 43.3 / 62.9|69.3 / 45.0 /62.6|
|BEA19-test|ERRANT (P/R/F0.5)|67.7 / 50.1 / 63.3|70.4 / 55.0 / 66.6|68.3 / 57.1 /65.6|
|JFLEG-test|GLEU|55.2|57.8|57.3|

The above results are for a single model. The beam width was set to 5 in the inference.  

You can use these models by:
```
python generate.py \
 --restore_dir gotutiyan/gec-bart-base \
 --input <a raw text file that you want to correct errors> \
 > output_file.txt
```
