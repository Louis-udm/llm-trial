# -*- coding: utf-8 -*-

# # # #
# offensive_bert.py
# @author Zhibin.LU
# @created Mon Mar 04 2019 11:32:27 GMT-0500 (EST)
# @last-modified Mon Apr 22 2019 09:02:12 GMT-0400 (EDT)
# @website: https://louis-udm.github.io
# @description
# # # #

#%%
'''
Use BERT Model
Data is directly the sentences with text of emoji.
'''

# %%
import sys
import os
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import gc

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import gc

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils import data

# from tqdm import tqdm, trange
import collections
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel, BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam #, warmup_linear

def set_work_dir(local_path="stage/VGCN-BERT", server_path="myprojects/VGCN-BERT"):
    if os.path.exists(os.path.join(os.getenv("HOME"), local_path)):
        os.chdir(os.path.join(os.getenv("HOME"), local_path))
    elif os.path.exists(os.path.join(os.getenv("HOME"), server_path)):
        os.chdir(os.path.join(os.getenv("HOME"), server_path))
    else:
        raise Exception('Set work path error!')

set_work_dir()
print('Current host:',os.uname()[1],'Current dir:', os.getcwd())

import random
random.seed(44)
np.random.seed(44)
torch.manual_seed(44)

cuda_yes = torch.cuda.is_available()
if cuda_yes:
    torch.cuda.manual_seed_all(44)
# cuda_yes = False
# print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")
# print('Device:',device)

#%%
'''
Configuration
'''

# will_train_mode_from_checkpoint=False
# n10fold=0
# cfg_ds='hate'
# cfg_stop_words=False

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='mr')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--sw', type=int, default='0')
args = parser.parse_args()
cfg_stop_words=True if args.sw==1 else False
will_train_mode_from_checkpoint=True if args.load==1 else False
cfg_ds = args.ds
n10fold=args.fold

dataset_list={'sst', 'hate', 'mr', 'cola', 'arhate'}

if cfg_ds=='R8':
    cfg_stop_words=True

will_train_mode=True
# will_train_mode_from_checkpoint=False
total_train_epochs = 9
# batch_size = 16
# dropout_rate = 0.2  #0.5 # Dropout rate (1 - keep probability).
# learning_rate0 = 1e-5 #2e-5   # "The initial learning rate for Adam."
# batch_size = 16 #16 #32 
# l2_decay = 0.01 #default,data0
if cfg_ds=='hate':
    batch_size = 12 #12
    learning_rate0 = 4e-6 #2e-5
    l2_decay = 2e-4 #data3
elif cfg_ds =='sst':
    batch_size = 16 #12
    learning_rate0 = 1e-5 #2e-5  
    # l2_decay = 0.001 
    l2_decay = 0.01 #default
elif cfg_ds=='cola':
    batch_size = 16 #12
    learning_rate0 = 8e-6 #2e-5  
    l2_decay = 0.01 
elif cfg_ds=='arhate':
    total_train_epochs = 9
    batch_size = 16 #12
    learning_rate0 = 1e-5  
    l2_decay = 0.001
elif cfg_ds=='mr':
    batch_size = 16 #12   
    learning_rate0 = 8e-6 #2e-5  
    l2_decay = 0.01
    # l2_decay = 1e-4 #default
elif cfg_ds=='dahate':
    batch_size = 16 #12
    # learning_rate0 = 1.1e-5 #2e-5  
    learning_rate0 = 1e-5 #2e-5  
    l2_decay = 0.01

#(max_seq=256, batch_size=16, max_seq=128, batch_size=32)
MAX_SEQ_LENGTH = 200 #512 #256 #128 #300
gradient_accumulation_steps = 1
bert_model_scale = 'bert-base-uncased'
do_lower_case = True
# eval_batch_size = 8
# predict_batch_size = 8
# "Proportion of training to perform linear learning rate warmup for. "
# "E.g., 0.1 = 10% of training."
warmup_proportion = 0.1
# "How often to save the model checkpoint."
# save_checkpoints_steps = 1000
# "How many steps to make in each estimator call."
# iterations_per_loop = 1000
valid_data_taux = 0.15 #1.0/6.0 #0.12
# "1-stem_examples, 2-basic_examples, 3-emotion_basic_examples"
output_dir = './output/'

# read_dump_objects=True
if cfg_stop_words:
    data_dir='data_gcn'
else:
    data_dir='data_gcn_allwords'
output_dir = './output/'

# perform_metrics_str=['macro avg','recall']
# perform_metrics_str=['micro avg','recall']
perform_metrics_str=['weighted avg','f1-score']


# cfg_ds=0  #*
# 选择了data set后，决定train data，保存model，和evaluat performance用的test/valid data
# 有confidence(比如have_offensiev data set)的时候，使用mse+weighted resample
# 没有confidence的时候，使用cross entropy loss + weighted loss
# resample_train_set=False # if mse and resample, then do resample
do_softmax_before_mse=True
# HATE_OFFENSIVE and LARGETWEETS
if cfg_ds in ('hate','dahate'):
    cfg_loss_criterion = 'mse'
else:
    cfg_loss_criterion='cle'
if n10fold>0:
    model_file_save='bert_model_'+cfg_ds+'_'+cfg_loss_criterion+'_'+"sw"+str(int(cfg_stop_words))+'_'+str(n10fold)+'_compare2.pt'
else:
    model_file_save='bert_model_'+cfg_ds+'_'+cfg_loss_criterion+'_'+"sw"+str(int(cfg_stop_words))+'_compare2.pt'


# model_file_evaluate 只决定做evaluat performance是用什么model file来装载
# 当evaluate_model_choice==cfg_ds时，使用相同数据训练获得的model测试相同数据集
model_file_evaluate=model_file_save
# model_file_evaluate='off_bert_model_HATE_OFFENSIVE_mse_0.pt'
# model_file_evaluate='offensive_coling_bert_model_lw_0.pt' 
print('Bert Start at:', time.asctime())
print('\n*** Configure: ***',
    '\n  cfg_ds:',cfg_ds,
    '\n  stop_words:',cfg_stop_words,
    '\n  MAX_SEQ_LENGTH:',MAX_SEQ_LENGTH,'valid_data_taux',valid_data_taux,
    '\n  learning_rate',learning_rate0,'weight_decay',l2_decay,
    '\n  Loss_criterion',cfg_loss_criterion,'softmax_before_mse',do_softmax_before_mse,
    '\n  perform_metrics_str:',perform_metrics_str,
    '\n  model_file_evaluate:',model_file_evaluate)


#%%

'''
Functions and Classes for read and organize data set
'''
class InputExample(object):
    '''
    A single training/test example for sentence classifier.
    '''

    def __init__(self, guid, text_a, text_b=None, confidence=None, label=None):
        '''
        Constructs a InputExample.

        Args:
            guid: Unique id for the example(a sentence or a pair of sentences).
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        '''
        self.guid = guid
        # string of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.text_a = text_a
        self.text_b = text_b
        # the label(class) for the sentence
        self.confidence=confidence
        self.label = label


class InputFeatures(object):
    '''
    A single set of features of data.
    result of convert_examples_to_features(InputExample)

    *** Example ***
    guid: 0 ,编号为0的句子对
    tokens(是list): [CLS] who was jim henson ? [SEP] jim henson was a puppet ##eer [SEP]
    input_ids(token的索引): 101([CLS]) 2040 2001 3958 27227 1029 102 3958 27227 2001 1037 13997 11510 102 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    input_mask(attention_mask, 句子长度小于max_length时，0是剩下的padding,或叫attention_mask): 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0
    segment_ids(第一句话0第二句话1, 或叫token_type_ids): 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    segment_ids:如果是两个句子，则sentence1先和sentene2连接后，两句子加上1个[CLS]和两个[SEP]的总长度不足最长max_seq_len的时候再补0.
    predict_mask: [CLS],[SEP],以及对于被Bert分词分出来的 '##xxx'类型的token, NER label都用'X'代替(见bert paper), 以及padding的部分也是X, 并且'X'的predict_mask也是0
    最后输出计算正确率的时候，使用predict_mask=0屏蔽[CLS],[SEP],X这三个假NER标签: 0 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    label_ids: [9([CLS]),1,2,5,0,11,11,11,11(X),0,....,10([SEP]),11(X),11,11,11,11,11,11,11,11,11]
    注意，做NER任务的时候InputExample中的sentence用bertTokenrizer进行转换后，得记住"##分词"的部分，不然会导致词和label不对。
    '''

    def __init__(self, guid, tokens, input_ids, input_mask, segment_ids, confidence, label_id):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.confidence=confidence
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    '''
    Truncates a sequence pair in place to the maximum length.
    '''

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def example2feature(example, tokenizer):

    # tokens_a = tokenizer.tokenize(example.text_a)
    # 因为已经在text_gcn/off_data_prepare.py中用bert.tokenizer分过。
    tokens_a = example.text_a.split()
    tokens_b = None
    if example.text_b:
        # tokens_b = tokenizer.tokenize(example.text_b)
        tokens_b = example.text_b.split()
        # 防止句子过长
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        # 如果tokens_a和tokens_b的总长度(和)超出MAX_SEQ_LENGTH-3，就对他们进行truncate
        if len(tokens_a) + len(tokens_b)>MAX_SEQ_LENGTH - 2:
            print('GUID: %d, Sentence a+b are too long: %d, %d'%(example.guid, len(tokens_a),len(tokens_b)))
            _truncate_seq_pair(tokens_a, tokens_b, MAX_SEQ_LENGTH - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > MAX_SEQ_LENGTH - 2:
            print('GUID: %d, Sentence is too long: %d'%(example.guid, len(tokens_a)))
            tokens_a = tokens_a[:(MAX_SEQ_LENGTH - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    feat=InputFeatures(
            guid=example.guid,
            tokens=tokens,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            # label_id=label2idx[example.label]
            confidence=example.confidence,
            label_id=example.label
    )
    return feat

class CorpusDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples=examples
        self.tokenizer=tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat=example2feature(self.examples[idx], self.tokenizer)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.confidence, feat.label_id

    @classmethod
    def pad(cls,batch):
        # 这个sample指__getitem__返回的内容
        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f_collect = lambda x: [sample[x] for sample in batch]
        f_pad = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
        # batch_input_ids = torch.LongTensor(f_pad(0, maxlen))
        # batch_input_mask = torch.LongTensor(f_pad(1, maxlen))
        # batch_segment_ids = torch.LongTensor(f_pad(2, maxlen))
        # # batch_predict_mask = torch.ByteTensor(f_pad(3, maxlen))
        # batch_confidence = torch.FloatTensor(f_collect(3))
        # batch_label_id = torch.LongTensor(f_collect(4))

        # torch.tensor会从data中的数据部分做拷贝（而不是直接引用），
        # 根据原始数据类型生成相应的torch.LongTensor、torch.FloatTensor和torch.DoubleTensor。
        batch_input_ids = torch.tensor(f_pad(0, maxlen), dtype=torch.long)
        batch_input_mask = torch.tensor(f_pad(1, maxlen), dtype=torch.long)
        batch_segment_ids = torch.tensor(f_pad(2, maxlen), dtype=torch.long)
        batch_confidences = torch.tensor(f_collect(3), dtype=torch.float)
        batch_label_ids = torch.tensor(f_collect(4), dtype=torch.long)

        return batch_input_ids, batch_input_mask, batch_segment_ids, batch_confidences, batch_label_ids


#%%
'''
Prepare data set
'''

import pickle as pkl

objects=[]
if n10fold<=0:
    names = [ 'labels','train_y','train_y_prob', 'valid_y','valid_y_prob','test_y','test_y_prob', 'shuffled_clean_docs']
    for i in range(len(names)):
        datafile="./"+data_dir+"/data_%s.%s"%(cfg_ds,names[i])
        with open(datafile, 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    lables_list,train_y, train_y_prob,valid_y,valid_y_prob,test_y,test_y_prob, shuffled_clean_docs=tuple(objects)

else:
    names = [ 'labels', 'gen_y','gen_y_prob']
    for i in range(len(names)):
        datafile="./"+data_dir+"/data_%s.%s"%(cfg_ds,names[i])
        with open(datafile, 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    names_fold = [ 'train_y','train_y_prob', 'valid_y','valid_y_prob','test_y','test_y_prob', 'shuffled_clean_docs']
    for i in range(len(names_fold)):
        datafile="./"+data_dir+"/data_%s.fold%d.%s"%(cfg_ds,n10fold,names_fold[i])
        with open(datafile, 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    lables_list, gen_y, gen_y_prob, train_y, train_y_prob,valid_y,valid_y_prob,test_y,test_y_prob, shuffled_clean_docs=tuple(objects)

label2idx=lables_list[0]
idx2label=lables_list[1]
y=np.hstack((train_y,valid_y,test_y))
y_prob=np.vstack((train_y_prob,valid_y_prob,test_y_prob))

examples=[]
for i,ts in enumerate(shuffled_clean_docs):
    ex=InputExample(i, ts.strip(), confidence=y_prob[i],label=y[i])
    examples.append(ex)

num_classes=len(label2idx)
train_size = len(train_y)
valid_size = len(valid_y)
test_size = len(test_y)

indexs = np.arange(0, len(examples))
train_examples = [examples[i] for i in indexs[:train_size]]
valid_examples = [examples[i] for i in indexs[train_size:train_size+valid_size]]
test_examples = [examples[i] for i in indexs[train_size+valid_size:train_size+valid_size+test_size]]

gc.collect()

# 这个balance的class_weight从sklearn中学来, 这个weight*propotion=1/num_class
# n_samples / (n_classes * np.bincount(y))
def get_class_count_and_weight(y,n_classes):
    classes_count=[]
    weight=[]
    for i in range(n_classes):
        count=np.sum(y==i)
        classes_count.append(count)
        weight.append(len(y)/(n_classes*count))
    return classes_count,weight

train_classes_num, train_classes_weight = get_class_count_and_weight(train_y,len(label2idx))

tokenizer = BertTokenizer.from_pretrained(bert_model_scale, do_lower_case=do_lower_case)

#%%

def get_pytorch_dataloader(examples, tokenizer, batch_size, shuffle_choice, classes_weight=None, total_resample_size=-1):
    dataset = CorpusDataset(examples, tokenizer)
    if shuffle_choice==0: # shuffle==False
        return DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=CorpusDataset.pad)
    elif shuffle_choice==1: # shuffle==True
        return DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=CorpusDataset.pad)
    elif shuffle_choice==2: #weighted resampled
        assert classes_weight is not None
        assert total_resample_size>0
        # weights = [13.4 if label == 0 else 4.6 if label == 2 else 1.0 for _,_,_,label in data]
        weights = [classes_weight[0] if label == 0 else classes_weight[1] if label == 1 else classes_weight[2] for _,_,_,_,label in dataset]
        sampler = WeightedRandomSampler(weights, num_samples=total_resample_size, replacement=True)
        return DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=4,
                                collate_fn=CorpusDataset.pad)

                                
loss_weight=torch.tensor(train_classes_weight).to(device)
# total_resample_size = len(train_label_ids)
# total_resample_size = int(len(label2idx)*np.max(train_classes_num))
# if cfg_loss_criterion=='mse' and resample_train_set:
#     train_dataloader = get_pytorch_dataloader(train_examples, tokenizer, batch_size, shuffle_choice=2,
#         classes_weight=train_classes_weight, total_resample_size=total_resample_size)
# else:
train_dataloader = get_pytorch_dataloader(train_examples, tokenizer, batch_size, shuffle_choice=0 )
valid_dataloader = get_pytorch_dataloader(valid_examples, tokenizer, batch_size, shuffle_choice=0 )
test_dataloader = get_pytorch_dataloader(test_examples, tokenizer, batch_size, shuffle_choice=0 )


# total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)
total_train_steps = int(len(train_dataloader) / gradient_accumulation_steps * total_train_epochs)

print("***** Running training *****")
print('  Train_classes count:', train_classes_num)
print('  Num examples for train =',len(train_examples),', after weight sample:',len(train_dataloader)*batch_size)
print("  Num examples for validate = %d"% len(valid_examples))
print("  Batch size = %d"% batch_size)
print("  Num steps = %d"% total_train_steps)

# for aa in valid_dataloader:
#     print(aa[0,0].shape,aa[0,1].shape,aa[0,2].shape)
#     break
#%%
'''
BertForTextClassification, only for one document classify
modify weighted loss from BertForSequenceClassification
'''
class BertForTextClassification(BertForSequenceClassification):

    def __init__(self, config, num_labels,output_attentions):
        super(BertForTextClassification, self).__init__(config, num_labels,output_attentions=output_attentions)
        
        self.will_collect_cls_states=False
        self.all_cls_states=[]
        self.output_attentions=output_attentions

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, loss_weight=None, confidence=None):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # 这个分类用的pooled_output其实是bertModel中的12层hidden_layer中的最后一层的[CLS]的hidden_status的vector
        # 也就是bert的分类,最后用的只是CLS  
        if self.output_attentions:
            all_attentions, _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        else:      
            _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        if self.will_collect_cls_states:
            self.all_cls_states.append(pooled_output.cpu())

        pooled_output = self.dropout(pooled_output)
        score_out = self.classifier(pooled_output)
        # if self.training:
        #     self.bert.train()
        #     _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # else:
        #     self.bert.eval()
        #     with torch.no_grad():
        #         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # pooled_output = self.dropout(pooled_output)
        # score_out = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions,score_out
        return score_out

def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    # print("***** Running prediction *****")
    model.eval()
    predict_out = []
    all_label_ids = []
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, y_prob, label_ids = batch
            # the parameter label_ids is None, model return the prediction score
            logits = model(input_ids, segment_ids, input_mask)

            if cfg_loss_criterion=='mse':
                if do_softmax_before_mse:
                    logits=F.softmax(logits,-1)
                # mse的时候相当于对输出概率同原始概率做mse的拟合
                loss = F.mse_loss(logits, y_prob)
            else:
                # cross entropy loss中已经包含softmax
                if loss_weight is None:
                    loss = F.cross_entropy(logits.view(-1, num_classes), label_ids)
                else:
                    loss = F.cross_entropy(logits.view(-1, num_classes), label_ids)
            # ev_loss+=loss.item()

            _, predicted = torch.max(logits, -1)
            predict_out.extend(predicted.tolist())
            all_label_ids.extend(label_ids.tolist())
            # eval_accuracy = accuracy(out_scores.detach().cpu().numpy(), label_ids.to('cpu').numpy())
            eval_accuracy=predicted.eq(label_ids).sum().item()
            total += len(label_ids)
            correct += eval_accuracy

        f1_metrics=f1_score(np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1), average='weighted')
        perform_metrics = classification_report(np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),output_dict=True)[perform_metrics_str[0]][perform_metrics_str[1]]
        print("Report:\n"+classification_report(np.array(all_label_ids).reshape(-1),
            np.array(predict_out).reshape(-1),digits=4))
        print(perform_metrics, f1_metrics)

    test_acc = correct/total
    end = time.time()
    print('Epoch : %d, %s: %.3f Acc : %.3f on %s, Spend:%.3f minutes for evaluation'
        % (epoch_th, ' '.join(perform_metrics_str), 100*perform_metrics, 100.*test_acc, dataset_name,(end-start)/60.0))
    print('--------------------------------------------------------------')
    return test_acc, perform_metrics


'''
train model
'''
if will_train_mode:
    if will_train_mode_from_checkpoint and os.path.exists(os.path.join(output_dir, model_file_save)):
        checkpoint = torch.load(os.path.join(output_dir, model_file_save), map_location='cpu')
        start_epoch = checkpoint['epoch']+1
        valid_acc_prev = checkpoint['valid_acc']
        perform_metrics_prev = checkpoint['perform_metrics']
        # do_lower_case = checkpoint['do_lower_case']
        model = BertForTextClassification.from_pretrained(bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label2idx),output_attentions=False)
        # model = BertForSequenceClassification.from_pretrained(bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label2idx))
        # model = BertForSequenceClassification.from_pretrained('bert-large-uncased', state_dict=checkpoint['model_state'], num_labels=len(label2idx))
        # BertForNER.from_pretrained(config['task']['bert_model_dir'], state_dict=checkpoint['model_state'],
        #                                        num_labels=len(label_list))
        print('Loaded the pretrain model:',model_file_save,', epoch:',checkpoint['epoch'],'valid acc:',
            checkpoint['valid_acc'],' '.join(perform_metrics_str)+'_valid:', checkpoint['perform_metrics'])

    else:
        start_epoch = 0
        valid_acc_prev = 0
        perform_metrics_prev = 0
        model = BertForTextClassification.from_pretrained(bert_model_scale, num_labels=len(label2idx),output_attentions=False)
        # model = BertForSequenceClassification.from_pretrained(bert_model_scale, num_labels=len(label2idx))
        # model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=len(label2idx))
 
    model.to(device)

    # Prepare optimizer
    # named_params = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': l2_decay},
    #     {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                 lr=learning_rate0,
    #                 warmup=warmup_proportion,
    #                 t_total=total_train_steps)

    # other_lr=1e-3
    # bert_lr=1e-6
    # all_params = set(model.parameters())
    # bert_params = set(model.bert.parameters())
    # other_params = all_params - bert_params
    # params = [{"params": list(bert_params), "lr": bert_lr},
    #     {"params": list(other_params), "lr": other_lr},]
    # # optimizer = optim.Adam(params)
    # optimizer = BertAdam(params, warmup=warmup_proportion, t_total=total_train_steps, weight_decay=l2_decay)

    optimizer = BertAdam(model.parameters(), lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps, weight_decay=l2_decay)

    # train using only BertForTokenClassification
    train_start = time.time()
    global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)
    # for epoch in trange(start_epoch, total_train_epochs, desc="Epoch"):
    # loss_weight=torch.tensor(train_classes_weight).to(device)
    for epoch in range(start_epoch, total_train_epochs):
        tr_loss = 0
        ep_train_start = time.time()
        model.train()
        optimizer.zero_grad()
        # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, confidence, label_ids = batch

            # the parameter label_ids is not None, model return the crossEntropyLoss
            # loss = model(input_ids, segment_ids, input_mask, label_ids)
            # mse的时候使用人类评判的比例作为confidence,并使用weighted resampler
            # loss_criterion='mse' or 'cel',
            # cel, cross entropy loss 使用weighted loss
            logits = model(input_ids, segment_ids, input_mask)
                    # label_ids, loss_weight, loss_criterion=cfg_loss_criterion, confidence=confidence)
                    # label_ids, loss_criterion=cfg_loss_criterion, confidence=confidence)

            if cfg_loss_criterion=='mse':
                if do_softmax_before_mse:
                    logits=F.softmax(logits,-1)
                # mse的时候相当于对输出概率同原始概率做mse的拟合
                loss = F.mse_loss(logits, confidence)
            else:
                # cross entropy loss中已经包含softmax
                if loss_weight is None:
                    loss = F.cross_entropy(logits, label_ids)
                else:
                    loss = F.cross_entropy(logits.view(-1, num_classes), label_ids, loss_weight)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            loss.backward()

            tr_loss += loss.item()
            # gradient_accumulation_steps=1时，每个mini batch都会更新
            if (step + 1) % gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                # lr_this_step = learning_rate0 * warmup_linear(global_step_th/total_train_steps, warmup_proportion)
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1
            if step % 40 == 0:
                print("Epoch:{}-{}/{}, Train {}: {} ".format(epoch, step, len(train_dataloader), cfg_loss_criterion,loss.item()))

        print('--------------------------------------------------------------')
        print("Epoch:{} completed, Total Train Loss:{}, Spend {}m ".format(epoch, tr_loss,(time.time() - train_start)/60.0))
        valid_acc,perform_metrics = evaluate(model, valid_dataloader, batch_size, epoch, 'Valid_set')
        _,test_f1 = evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')
        # Save a checkpoint
        # if valid_acc > valid_acc_prev:
        if perform_metrics > perform_metrics_prev:
            # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            to_save={'epoch': epoch, 'model_state': model.state_dict(),
                        'valid_acc': valid_acc, 'lower_case': do_lower_case,
                        'perform_metrics':perform_metrics}
            torch.save(to_save, os.path.join(output_dir, model_file_save))
            # valid_acc_prev = valid_acc
            perform_metrics_prev = perform_metrics
            test_f1_when_valid_best=test_f1
            valid_f1_best_epoch=epoch

    print('\n**Optimization Finished!,Total spend:',(time.time() - train_start)/60.0)
    print("**Valid weighted F1: %.3f at %d epoch."%(100*perform_metrics_prev,valid_f1_best_epoch))
    print("**Test weighted F1 when valid best: %.3f"%(100*test_f1_when_valid_best))

    valid_f1=perform_metrics_prev*100
    test_f1=test_f1_when_valid_best*100
    if n10fold>0:
        fold_file="./"+data_dir+"/bert_data_%s_%s.fold_log"%(cfg_ds,"sw"+str(args.sw))
        if n10fold==1:
            with open(fold_file, 'wb') as f:
                pkl.dump({'valid':[valid_f1],'test':[test_f1]}, f)
        else:
            with open(fold_file, 'rb') as f:
                fold_log=pkl.load(f, encoding='latin1')
            fold_log['valid'].append(valid_f1)
            fold_log['test'].append(test_f1)
            with open(fold_file, 'wb') as f:
                pkl.dump(fold_log, f)
        if n10fold==10:
            print('10 fold list:',fold_log)
            print('10 fold result: valid_f1=%.2f, test_f1=%.2f'%(np.mean(fold_log['valid']),np.mean(fold_log['test'])))
    


#%%
'''
Evaluat performance, get report and confusion matrix
'''
import pandas as pd
import seaborn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def predict(model, examples, tokenizer, batch_size):
    dataloader=get_pytorch_dataloader(examples, tokenizer, batch_size, shuffle_choice=0)
    predict_out = []
    confidence_out=[]
    # all_label_ids=[]
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, _, label_ids = batch
            if model.output_attentions:
                _,score_out = model(input_ids, segment_ids, input_mask)
            else:
                score_out = model(input_ids, segment_ids, input_mask)
            if cfg_loss_criterion=='mse' and do_softmax_before_mse:
                score_out=torch.nn.functional.softmax(score_out,dim=-1)
            predict_out.extend(score_out.max(1)[1].tolist())
            confidence_out.extend(score_out.max(1)[0].tolist())
            
    #         all_label_ids.extend(label_ids.tolist())
    # print("*******Report:\n"+classification_report(np.array(all_label_ids).reshape(-1),
    #     np.array(predict_out).reshape(-1),digits=4))
    return np.array(predict_out).reshape(-1), np.array(confidence_out).reshape(-1)

def plt_conf_matrix(conf_matrix,metrics_name,metrics_score,outfile_suffix):
    # names=['Hate','Offensive','Neither']
    names=label2idx.keys()
    confusion_df = pd.DataFrame(conf_matrix, index=names,columns=names)
    plt.figure(figsize=(5,5))
    seaborn.heatmap(confusion_df,annot=True,annot_kws={"size": 12},cmap='gist_gray_r',cbar=False, square=True,fmt='.2f')
    plt.ylabel(r'True categories',fontsize=14)
    plt.xlabel(r'Predicted categories',fontsize=14)
    plt.title('BERT, ' + outfile_suffix + ', '+metrics_name+': %.2f%%'%metrics_score)
    plt.tick_params(labelsize=12)
    plt.savefig('figure/off_conf_matrix_bert_'+outfile_suffix+'.pdf')
    plt.show()

def print_report(y_true, y_preds, outfile_suffix):
    metrics_name=perform_metrics_str[0]+' '+perform_metrics_str[1]
    print('\n\nModel:',model_file_evaluate,', Data set:',outfile_suffix)
    report = 'Evaluate report:\n'+classification_report(y_true, y_preds, digits=4)
    print(report)
    # metrics_score=(100*np.sum(y_preds==y_true)/len(y_true))
    metrics_score=f1_score(y_true, y_preds, average='weighted')
    print("weighted f1-score:", metrics_score)
    # same
    # print(classification_report(y_true, y_preds, output_dict=True)[perform_metrics_str[0]][perform_metrics_str[1]])

    conf_matrix = confusion_matrix(y_true,y_preds)
    matrix_proportions = np.zeros((len(label2idx),len(label2idx)))
    for i in range(len(label2idx)):
        matrix_proportions[i,:] = conf_matrix[i,:]/float(conf_matrix[i,:].sum())
    print('Confusion matrix:\n',matrix_proportions)
    # plt_conf_matrix(matrix_proportions,metrics_name,metrics_score,outfile_suffix)



#%%

do_statistic=False

'''
print the errors predictions
'''
def get_dataframe_with_predict(model, examples, y_true):
    y_preds,y_pred_confid=predict(model, examples, tokenizer, batch_size)
    # err_pred_dict={}
    df=pd.DataFrame(columns=['guid', 'label', 'pred', 'pred_confid', 'sent'])
    for i in range(len(y_true)):
        df=df.append({'guid':examples[i].guid,'label':y_true[i],'pred':y_preds[i],'pred_confid':y_pred_confid[i],'sent':examples[i].text_a},ignore_index=True)
        # if y_preds[i]!=y_true[i]:
            # print('i:',i,'Guid:',examples[i].guid,'Label:',y_true[i],
            #     'Predict:',y_preds[i], 'Confid:',y_pred_confid[i], 'token:',
            #        ' '.join(examples[i].tokens),'sentence:',examples[i].text_a)
            # err_pred_dict[examples[i].guid]=[y_true[i],y_preds[i],examples[i].text_a]
    # return err_pred_dict,df
    return df


if do_statistic:
    start=time.time()

    # if os.path.exists(os.path.join(output_dir, save_file_name):
    # model_file_evaluate='output/off_bert_model_LARGETWEETS_mse_sw0_lr6e-6_compare.pt'
    checkpoint = torch.load(os.path.join(output_dir, model_file_evaluate), map_location='cpu')
    model = BertForTextClassification.from_pretrained(bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label2idx),output_attentions=False)
    print('Loaded the pretrain Bert model:',model_file_evaluate,', epoch:',checkpoint['epoch'],'valid acc:',
        checkpoint['valid_acc'], ' '.join(perform_metrics_str)+'_valid:', checkpoint['perform_metrics'])
    model.to(device)
    model.eval()

    # valid_y_preds,_=predict(model, valid_examples, tokenizer, batch_size)
    # print_report(valid_y,valid_y_preds,cfg_ds+'_valid')
    # test_y_preds,_=predict(model, test_examples, tokenizer, batch_size)
    # print_report(test_y,test_y_preds,cfg_ds+'_test')

    # if cfg_ds==12:
    #     gen_y_preds,_=predict(model, gen_examples, tokenizer, batch_size)
    #     print_report(gen_y,gen_y_preds,cfg_ds+'_gen')

    print(model_file_evaluate+' output all the error classified sentences for valid_data:')
    mixed_df_dev_bert=get_dataframe_with_predict(model, valid_examples, valid_y)
    mixed_df_dev_bert.to_csv(output_dir+cfg_ds+'_bert_valid.csv')
    print('Use %.2f minutes to predict %d samples.'%((time.time()-start)/60,len(valid_examples)))

    start=time.time()
    print(model_file_evaluate+' output all the error classified sentences for test_data:')
    mixed_df_train_bert=get_dataframe_with_predict(model, test_examples, test_y)
    mixed_df_train_bert.to_csv(output_dir+cfg_ds+'_bert_test.csv')
    print('Use %.2f minutes to predict %d samples.'%((time.time()-start)/60,len(train_examples)))


#%%
# # test
# # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# vocab1=tokenizer.vocab
# vocab2=prepare.vocab
# tt=set()
# for seq in prepare.basic_tokens:
#     for t in seq:
#         if t not in vocab1:
#             tt.add(t)
# #%%
# seqs=[' '.join(ts).strip() for ts in prepare.basic_tokens]
# #%%
# ttt=[]
# for s in seqs:
#     ttt.append(tokenizer.tokenize(s))

# #%%

# max_len=0
# max_len_idx=0
# for i,seq in enumerate(ttt):
#   if len(seq)>max_len:
#     max_len=len(seq)
#     max_len_idx=i
