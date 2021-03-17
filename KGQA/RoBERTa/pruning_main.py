import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import operator
from torch.nn import functional as F
from pruning_dataloader import DatasetPruning, DataLoaderPruning
from pruning_model import PruningModel
from torch.optim.lr_scheduler import ExponentialLR
import networkx as nx
from collections import defaultdict



parser = argparse.ArgumentParser()


parser.add_argument('--ls', type=float, default=0.1)
parser.add_argument('--validate_every', type=int, default=25)

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=90)
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--patience', type=int, default=15)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
args = parser.parse_args()

def printRelationText(rel_ids, idx2rel):
    rel_text = []
    for r in rel_ids:
        if r not in idx2rel:
            r = r.item()
        rel_text.append(idx2rel[r])
    print(rel_text)

def validate_v2(model, device, train_dataset, rel2idx, idx2rel):
    model.eval()
    data = process_data_file('pruning_test.txt', rel2idx, idx2rel)
    num_correct = 0
    count = 0
    for i in tqdm(range(len(data))):
        # try:
        d = data[i]
        question = d[0]
        question_tokenized, attention_mask = train_dataset.tokenize_question(question)
        question_tokenized = question_tokenized.to(device)
        attention_mask = attention_mask.to(device)
        rel_id_list = d[1]
        scores = model.get_score_ranked(question_tokenized=question_tokenized, attention_mask=attention_mask)
        top2 = torch.topk(scores, 1)
        top2 = top2[1]
        isCorrect = False
        for x in top2:
            if x in rel_id_list:
                isCorrect = True
        if isCorrect:
            num_correct += 1
        # else:
        #     print(d[2])
        #     printRelationText(top2, idx2rel)
        #     printRelationText(rel_id_list, idx2rel)
        #     count += 1
        #     if count == 10:
        #         exit(0)
        # pred_rel_id = torch.argmax(scores).item()
        # if pred_rel_id in rel_id_list:
        #     num_correct += 1

            
    # np.save("scores_webqsp_complex.npy", scores_list)
    # exit(0)
    accuracy = num_correct/len(data)
    return accuracy

def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return

def process_data_file(fname, rel2idx, idx2rel):
    f = open(fname, 'r')
    data = []
    for line in f:
        line = line.strip().split('\t')
        question = line[0].strip()
        #TODO only work for webqsp. to remove entity from metaqa, use something else
        #remove entity from question
        question = question.split('[')[0]
        rel_list = line[1].split('|')
        rel_id_list = []
        for rel in rel_list:
            rel_id_list.append(rel2idx[rel])
        data.append((question, rel_id_list, line[0].strip()))
    return data

def train(batch_size, shuffle, num_workers, nb_epochs, gpu, use_cuda, patience, validate_every, lr, decay, ls):
    # f = open('/scratche/home/apoorv/mod_TuckER/models/ComplEx_fbwq_full/relations.dict', 'r')
    f = open('/scratche/home/apoorv/mod_TuckER/data/fbwq_full_allrels/relations.dict', 'r')
    rel2idx = {}
    idx2rel = {}
    for line in f:
        line = line.strip().split('\t')
        id = int(line[1])
        rel = line[0]
        rel2idx[rel] = id
        idx2rel[id] = rel
    f.close()
    data = process_data_file('pruning_train.txt', rel2idx, idx2rel)
    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetPruning(data=data, rel2idx = rel2idx, idx2rel = idx2rel)
    data_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = PruningModel(rel2idx, idx2rel, ls)
    # checkpoint_file = "checkpoints/pruning/best_best.pt"
    # checkpoint = torch.load(checkpoint_file)
    # model.load_state_dict(checkpoint)
    # print('loaded from ', checkpoint_file)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    for epoch in range(nb_epochs):
        phases = []
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    question_tokenized = a[0].to(device)
                    attention_mask = a[1].to(device)
                    rel_one_hot = a[2].to(device)
                    loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask, rel_one_hot=rel_one_hot)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()
                
                scheduler.step()

            elif phase=='valid':
                model.eval()
                eps = 0.0001
                score = validate_v2(model=model, device=device, train_dataset=dataset,rel2idx=rel2idx, idx2rel = idx2rel)
                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    best_model = model.state_dict()
                    print("Validation accuracy increased from previous epoch", score)
                    # writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
                    torch.save(model.state_dict(), "checkpoints/pruning/best_mar12_3.pt")
                elif (score < best_score + eps) and (no_update < patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                elif no_update == patience:
                    print("Model has exceed patience. Saving best model and exiting")
                    exit()
                if epoch == nb_epochs-1:
                    print("Final Epoch has reached. Stoping and saving model.")
                    exit()
                    



def data_generator(data, roberta_file, entity2idx):
    question_embeddings = np.load(roberta_file, allow_pickle=True)
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1]
        # encoded_question = question_embedding[question]
        encoded_question = question_embeddings.item().get(question)
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), torch.tensor(encoded_question), ans, data_sample[1]



train(
    batch_size=args.batch_size,
    shuffle=args.shuffle_data, 
    num_workers=args.num_workers,
    nb_epochs=args.nb_epochs, 
    gpu=args.gpu, 
    use_cuda=args.use_cuda, 
    patience=args.patience,
    validate_every=args.validate_every,
    lr=args.lr,
    decay=args.decay,
    ls=args.ls)
