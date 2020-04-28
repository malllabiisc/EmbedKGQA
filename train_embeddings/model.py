import numpy as np
import torch
from torch.nn.init import xavier_normal_
import torch.nn as nn
import torch.nn.functional as F

class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.model = kwargs["model"]
        multiplier = 3
        self.loss_type = kwargs['loss_type']

        if self.loss_type == 'BCE':
            # self.loss = torch.nn.BCELoss()
            self.loss = self.bce_loss
            self.bce_loss_loss = torch.nn.BCELoss()
        elif self.loss_type == 'CE':
            self.loss = self.ce_loss
        else:
            print('Incorrect loss specified:', self.loss_type)
            exit(0)
        if self.model == 'DistMult':
            multiplier = 1
            self.score_func = self.DistMult
        elif self.model == 'SimplE':
            multiplier = 2
            self.score_func = self.SimplE
        elif self.model == 'ComplEx':
            multiplier = 2
            self.score_func = self.ComplEx
        elif self.model == 'RESCAL':
            self.score_func = self.RESCAL
            multiplier = 1
        elif self.model == 'TuckER':
            self.score_func = self.TuckER
            multiplier = 1
            self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        else:
            print('Incorrect model specified:', self.model)
            exit(0)
        self.E = torch.nn.Embedding(len(d.entities), d1 * multiplier, padding_idx=0)
        
        if self.model == 'RESCAL':
            self.R = torch.nn.Embedding(len(d.relations), d1 * d1, padding_idx=0)
        elif self.model == 'TuckER':
            self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        else:
            self.R = torch.nn.Embedding(len(d.relations), d1 * multiplier, padding_idx=0)

        self.entity_dim = d1 * multiplier
        self.do_batch_norm = True
        if kwargs["do_batch_norm"] == False:
            print('Not doing batch norm')
            self.do_batch_norm = False
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.l3_reg = kwargs["l3_reg"]

        if self.model in ['DistMult', 'RESCAL', 'SimplE', 'TuckER']: 
            self.bn0 = torch.nn.BatchNorm1d(d1 * multiplier)
            self.bn1 = torch.nn.BatchNorm1d(d1 * multiplier)
            self.bn2 = torch.nn.BatchNorm1d(d1 * multiplier)
        else:
            self.bn0 = torch.nn.BatchNorm1d(multiplier)
            self.bn1 = torch.nn.BatchNorm1d(multiplier)
            self.bn2 = torch.nn.BatchNorm1d(multiplier)

        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        print('Model is', self.model)
        print(self.E.weight.nelement() * self.E.weight.element_size())
        print(self.E.weight.nelement())
        
    def freeze_entity_embeddings(self):
        self.E.weight.requires_grad = False
        print('Entity embeddings are frozen')

    def ce_loss(self, pred, true):
        pred = F.log_softmax(pred, dim=-1)
        true = true/true.size(-1)
        loss = -torch.sum(pred * true)
        return loss

    def bce_loss(self, pred, true):
        loss = self.bce_loss_loss(pred, true)
        #l3 regularization
        if self.l3_reg:
            norm = torch.norm(self.E.weight.data, p=3, dim=-1)
            loss += self.l3_reg * torch.sum(norm)
        return loss

    def init(self):
        xavier_normal_(self.E.weight.data)            
        if self.model == 'Rotat3':
            nn.init.uniform_(self.R.weight.data, a=-1.0, b=1.0)
        else:
            xavier_normal_(self.R.weight.data)

    def RESCAL(self, head, relation):
        if self.do_batch_norm:
            head = self.bn0(head)
        head = self.input_dropout(head)
        head = head.view(-1, 1, self.entity_dim)
        relation = relation.view(-1, self.entity_dim, self.entity_dim)
        relation = self.hidden_dropout1(relation)
        x = torch.bmm(head, relation) 
        x = x.view(-1, self.entity_dim)  
        if self.do_batch_norm:    
            x = self.bn2(x)
        x = self.hidden_dropout2(x)
        s = torch.mm(x, self.E.weight.transpose(1,0))
        return s

    def TuckER(self, head, relation):
        if self.do_batch_norm:
            head = self.bn0(head)
        ent_embedding_size = head.size(1)
        head = self.input_dropout(head)
        head = head.view(-1, 1, ent_embedding_size)

        W_mat = torch.mm(relation, self.W.view(relation.size(1), -1))
        W_mat = W_mat.view(-1, ent_embedding_size, ent_embedding_size)
        W_mat = self.hidden_dropout1(W_mat)

        s = torch.bmm(head, W_mat) 
        s = s.view(-1, ent_embedding_size)      
        s = self.bn2(s)
        s = self.hidden_dropout2(s)
        s = torch.mm(s, self.E.weight.transpose(1,0))
        return s

    def DistMult(self, head, relation):
        if self.do_batch_norm:
            head = self.bn0(head)
        head = self.input_dropout(head)
        relation = self.hidden_dropout1(relation)
        s = head * relation
        if self.do_batch_norm:
            s = self.bn2(s)
        s = self.hidden_dropout2(s)
        s = torch.mm(s, self.E.weight.transpose(1,0))
        return s


    def SimplE(self, head, relation):
        if self.do_batch_norm:
            head = self.bn0(head)
        head = self.input_dropout(head)
        relation = self.hidden_dropout1(relation)
        s = head * relation
        s_head, s_tail = torch.chunk(s, 2, dim=1)
        s = torch.cat([s_tail, s_head], dim=1)
        if self.do_batch_norm:
            s = self.bn2(s)
        s = self.hidden_dropout2(s)
        s = torch.mm(s, self.E.weight.transpose(1,0))
        s = 0.5 * s
        return s

    def ComplEx(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.do_batch_norm:
            head = self.bn0(head)
        head = self.input_dropout(head)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        relation = self.hidden_dropout1(relation)
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.E.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.do_batch_norm:
            score = self.bn2(score)
        score = self.hidden_dropout2(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        return score


    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        h = e1
        r = self.R(r_idx)
        ans = self.score_func(h, r)
        pred = torch.sigmoid(ans)
        return pred
