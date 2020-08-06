# EmbedKGQA
This is the code for our ACL 2020 paper [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://malllabiisc.github.io/publications/papers/final_embedkgqa.pdf)

We will be updating the README with instructions on how to download the dataset and run the code.
![](model.png)

# Instructions

In order to run the code for MetaQA, first unzip data.zip and pretrained_model.zip

Then change to directory ./KGQA/LSTM. Following is an example command to run the QA training code

```
python3 main.py --mode train --relation_dim 200 --hidden_dim 256 \
--gpu 2 --freeze 0 --batch_size 128 --validate_every 5 --hops 2 --lr 0.0005 --entdrop 0.1 --reldrop 0.2  --scoredrop 0.2 \
--decay 1.0 --model ComplEx --patience 5 --ls 0.0 --kg_type half
```
# Dataset creation

## MetaQA

### KG dataset

There are 2 datasets: MetaQA_full and MetaQA_half. Full dataset contains the original kb.txt as train.txt with duplicate triples removed. Half contains only 50% of the triples (randomly selected without replacement). 

There are some lines like 'entity NOOP entity' in the train.txt for half dataset. This is because when removing the triples, all triples for that entity were removed, hence any KG embedding implementation would not find any embedding vector for them using the train.txt file. By including such 'NOOP' triples we are not including any additional information regarding them from the KG, it is there just so that we can directly use any embedding implementation to generate some random vector for them.

### QA Dataset

There are 5 files for each dataset (1, 2 and 3 hop)
- qa_train_nhop_train.txt
- qa_train_nhop_train_half.txt
- qa_train_nhop_train_old.txt
- qa_dev_nhop.txt
- qa_test_nhop.txt

Out of these, qa_dev, qa_test and qa_train_nhop_old are exactly the same as the MetaQA original dev, test and train files respectively.

For qa_train_nhop_train and qa_train_nhop_train_half, we have added triple (h, r, t) in the form of (head entity, question, answer). This is to prevent the model from 'forgetting' the entity embeddings when it is training the QA model using the QA dataset. qa_train.txt contains all triples, while qa_train_half.txt contains only triples from MetaQA_half.
