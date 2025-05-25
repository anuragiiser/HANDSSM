#import requirder modules
import os
import pandas as pd
import numpy as np
import random
import torch
import GPUtil
import csv
from tqdm import tqdm
from dataset import MyDataset , MyDataset2
from collections import Counter
import nltk
nltk.download('punkt', download_dir='/raid/home/dgx1382/students/PG/2022/amlan/download/nltk_data')
nltk.data.path.append('/raid/home/dgx1382/students/PG/2022/amlan/download/nltk_data')
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix, precision_score,recall_score 
import transformers
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
#read files and save a csv file 
import logging

import pickle5 as pickle
import sys
import csv

csv.field_size_limit(sys.maxsize)

#read files and save a csv file

logging.basicConfig(filename="HAN3.log",format='%(asctime)s \n %(message)s',filemode='w')
# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


list = {'no_class':0, 'yes_class':1}

def get_evaluation(y_true, y_prob, list_metrics):
    avg ='macro'
    y_pred = np.argmax(y_prob, -1)
    print(y_true, y_pred)
    output = {}
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(confusion_matrix(y_true, y_pred))
    if 'precision' in list_metrics:
        output['precision'] = precision_score(y_true, y_pred, average=avg)
    if 'recall' in list_metrics:
        output['recall'] = recall_score(y_true, y_pred, average=avg)
    if 'f1_score' in list_metrics:
        output['f1_score'] = f1_score(y_true, y_pred, average=avg)
    return output

def matrix_mul(input, weight, bias=False):
    feature_list = []
    for feature in input:
        feature = torch.mm(feature, weight)
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        feature = torch.tanh(feature).unsqueeze(0)
        feature_list.append(feature)

    return torch.cat(feature_list, 0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)

    return torch.sum(output, 0).unsqueeze(0)

def get_max_lengths(data_path):
    word_length_list = []
    sent_length_list = []
    with open(data_path) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        next(reader)
        for idx, line in enumerate(reader):
            query = line[0]
            statute = line[1]

            query_list = sent_tokenize(query)
            statute_list = sent_tokenize(statute)
            sent_length_list.append(len(query_list))
            sent_length_list.append(len(statute_list))

            for sent in query_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

            for sent in statute_list:
                word_list = word_tokenize(sent)
                word_length_list.append(len(word_list))

        sorted_word_length = sorted(word_length_list)
        sorted_sent_length = sorted(sent_length_list)

    #return sorted_word_length[int(0.8*len(sorted_word_length))], sorted_sent_length[int(0.8*len(sorted_sent_length))]
    return sorted_word_length[-1], sorted_sent_length[-1]

#parameters
word_hidden_size=50
sent_hidden_size=50
batch_size = 128


choice = 10

while(choice !=1 and choice !=2):
    print(choice)
    print('Choose the dataset you want to use:\n\t 1 for DATASET1 \n\t 2 for DATASET2')
    choice = int(input())
    print(choice)

if(choice ==1):
    train_set_path = 'train_1.csv'
    test_set_path = 'test_1.csv'
else:
    train_set_path = 'train_2.csv'
    test_set_path = 'test_2.csv'



max_word_length1, max_sent_length1 = get_max_lengths(train_set_path)
max_word_length2, max_sent_length2 = get_max_lengths(test_set_path)
max_word_length = max(max_word_length1, max_word_length2)
max_sent_length = max(max_sent_length1, max_sent_length2)
print('word_length',max_word_length)
print('sent_length', max_sent_length)
training_params = {"batch_size": batch_size,
                    "shuffle": True,
                    "drop_last": True}
test_params = {"batch_size": batch_size,
                "shuffle": False,
                "drop_last": False}
word2vec_path = 'glove.6B.50d.txt'
training_set = MyDataset2(train_set_path, word2vec_path, max_sent_length, max_word_length)
test_set = MyDataset2(test_set_path, word2vec_path, max_sent_length, max_word_length)
print(type(training_set))

training_generator = DataLoader(training_set, **training_params)
test_generator = DataLoader(test_set, **test_params)

def find_gpus(nums=6):
    os.system('nvidia-smi')
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_free_gpus')
    with open('tmp_free_gpus', 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [ (idx,int(x.split()[2]))
                              for idx,x in enumerate(frees) ]
    idx_freeMemory_pair.sort(key=lambda my_tuple:my_tuple[1],reverse=True)
    usingGPUs = [str(idx_memory_pair[0])
                    for idx_memory_pair in idx_freeMemory_pair[:nums] ]
    usingGPUs =  ','.join(usingGPUs)
    print('using GPU idx: #', usingGPUs)
    return usingGPUs

os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(nums=4)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


print(GPUtil.showUtilization())

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=2):
        super(SentAttNet, self).__init__()

        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))

        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
        # self.sent_softmax = nn.Softmax()
        # self.fc_softmax = nn.Softmax()
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        f_output, h_output = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim=0)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output
class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, hidden_size=50):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))

        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))

        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):

        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):

        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output, dim=0)
        output = element_wise_mul(f_output,output.permute(1,0))
        return output, h_output

class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, pretrained_word2vec_path,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(pretrained_word2vec_path, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state1 = torch.zeros(2, batch_size, self.word_hidden_size)
        self.word_hidden_state2 = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state1 = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.sent_hidden_state2 = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.hidden1 = nn.Linear(2*self.sent_hidden_size, 25)
        self.hidden2 = nn.Linear(25, 6)
        self.hidden3 = nn.Linear(6, 2)
        self.drop = nn.Dropout(0.25)
        self.softmax = nn.Softmax() 
        if torch.cuda.is_available():
            self.word_hidden_state1 = self.word_hidden_state1.cuda()
            self.sent_hidden_state1 = self.sent_hidden_state1.cuda()
            self.word_hidden_state2 = self.word_hidden_state2.cuda()
            self.sent_hidden_state2 = self.sent_hidden_state2.cuda()
            self.hidden1 = self.hidden1.cuda()
            self.hidden2 = self.hidden2.cuda()
            self.hidden3 = self.hidden3.cuda()
            self.drop = self.drop.cuda()
            self.softmax = self.softmax.cuda()
    def forward(self, query, statute):

        output_list1, output_list2 = [], []
        input1 = query.permute(1, 0, 2)
        for i in input1:
            output1, self.word_hidden_state1 = self.word_att_net(i.permute(1, 0), self.word_hidden_state1)
            output_list1.append(output1)
        output1 = torch.cat(output_list1, 0)
        output1, self.sent_hidden_state1 = self.sent_att_net(output1, self.sent_hidden_state2)

        input2 = statute.permute(1, 0, 2)
        for i in input2:
            output2, self.word_hidden_state2 = self.word_att_net(i.permute(1, 0), self.word_hidden_state2)
            output_list2.append(output2)
        output2 = torch.cat(output_list2, 0)
        output2, self.sent_hidden_state2 = self.sent_att_net(output2, self.sent_hidden_state2)
        

        self.sent_hidden_state1  = torch.mean(self.sent_hidden_state1, dim=0)
        self.sent_hidden_state2  = torch.mean(self.sent_hidden_state2, dim=0)
        #print((self.sent_hidden_state1).size(), (self.sent_hidden_state2).size())

        #out = torch.cosine_similarity(self.sent_hidden_state1, self.sent_hidden_state2)
        #print(out)
        #out  = torch.mean(out, dim=0)    #(torch.transpose(out, 0, 1))
        out = torch.cat((self.sent_hidden_state1 , self.sent_hidden_state2), dim=1)
        
        out = self.hidden1(out)
        out = self.drop(out)
        
        out = self.hidden2(out)
        out = self.drop(out)

        out = self.hidden3(out)
        #print(out)
        out = self.softmax(out)
        out = out.squeeze(1)
        #print(out)

        return out

model = HierAttNet(word_hidden_size, sent_hidden_size, batch_size, training_set.num_classes, word2vec_path, max_sent_length, max_word_length)

logger.info('model formed')

epoches = 10
#optimizer = AdamW(model.parameters(),lr = 1e-5,eps = 1e-8)
#scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps = len(dataloader_train)*epochs)

model.cuda()


writer = SummaryWriter('tensorboard/han_voc')
criterion = nn.CrossEntropyLoss()
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1, momentum=0.9)
best_loss = 1e5
best_epoch = 0
test_interval =1
es_min_delta = 0.0
es_patience = 5
model.train()
num_iter_per_epoch = len(training_generator)
for epoch in range(epoches):
    print('epoch',epoch)
    for iter, (query, statute, label) in enumerate(training_generator):
        print(iter)
        #print(label)
        query = query.cuda()
        statute = statute.cuda()
        label = label.cuda()
        #print(label)
        optimizer.zero_grad()
        #label = label.type(torch.LongTensor)
        model._init_hidden_state()
        predictions = model(query, statute)
        label = label.clamp(0, 1)
        #label = label.unsqueeze(1)
        #loss = criterion(predictions.float(), label.float())
        #print(predictions)
        #print(label)
        loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()
        training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
        logger.info("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
               epoch + 1,
            epoches,
            iter + 1,
            num_iter_per_epoch,
            optimizer.param_groups[0]['lr'],
            loss, training_metrics["accuracy"]))
        writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
        writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
    if epoch % test_interval == 0:
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for te_query, te_statute, te_label in test_generator:
            num_sample = len(te_label)
            te_query = te_query.cuda()
            te_statute = te_statute.cuda()
            te_label = te_label.cuda()
            with torch.no_grad():
                model._init_hidden_state(num_sample)
                te_predictions = model(te_query, te_statute)
            #te_label = te_label.unsqueeze(1)
            #print(te_predictions,te_label)
            te_loss = criterion(te_predictions, te_label)
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())
        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        #te_label = np.asarray(someListOfLists, dtype=np.float32)
        test_metrics = get_evaluation(te_label, te_predictions.cpu().detach().numpy(), list_metrics=["accuracy", "confusion_matrix"])
        logger.info(
            "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, opt.num_epoches,
                te_loss,
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            epoches,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        writer.add_scalar('Test/Loss', te_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        model.train()
        if te_loss + es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            #torch.save(model, opt.saved_path + os.sep + "whole_model_han")

        '''# Early stopping
        if epoch - best_epoch > es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
            break
            '''
        test_metrics = get_evaluation(te_label, te_pred.cpu(),
                              list_metrics=["accuracy", "loss", "confusion_matrix", "precision", "recall", "f1_score"])
        '''logger.info("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}\n Precision: {} Recall : {} F1_Score : {}".format(

                test_metrics["loss"],

                test_metrics["accuracy"],
                                                                            test_metrics["confusion_matrix"],

                test_metrics["precision"],

                test_metrics["recall"],

                test_metrics["f1_score"]))'''


model.eval()
te_label_ls = []
te_pred_ls = []
for te_query, te_statute, te_label in test_generator:
    num_sample = len(te_label)
    if torch.cuda.is_available():
        te_query = te_query.cuda()
        te_statute = te_statute.cuda()
        te_label = te_label.cuda()
    with torch.no_grad():
        model._init_hidden_state(num_sample)
        te_predictions = model(te_query, te_statute)
        #te_predictions = F.softmax(te_predictions, dim=0)
    te_label_ls.extend(te_label.clone().cpu())
    te_pred_ls.append(te_predictions.clone().cpu())
te_pred = torch.cat(te_pred_ls, 0).numpy()
te_label = te_label_ls


#for i, j in zip(te_label, te_pred):
#    print(f'True label: {i}, Predicted label: {np.argmax(j)}')

test_metrics = get_evaluation(te_label.cpu(), te_pred.cpu(),
                              list_metrics=["accuracy", "loss", "confusion_matrix", "precision", "recall", "f1_score"])
logger.info("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}\n Precision: {} Recall : {} F1_Score : {}".format(
                                                                            test_metrics["loss"],
                                                                            test_metrics["accuracy"],
                                                                            test_metrics["confusion_matrix"],
                                                                            test_metrics["precision"],
                                                                            test_metrics["recall"],
                                                                            test_metrics["f1_score"]))


logger.info('done')



'''
#import requirder modules
import pandas as pd
import numpy as np
import random
import torch
import GPUtil
import csv
import os
from tqdm import tqdm
from dataset import MyDataset
from collections import Counter
import nltk
import tensorflow as tf
nltk.download('punkt', download_dir='/raid/home/dgx1382/students/PG/2022/amlan/download/nltk_data')
nltk.data.path.append('/raid/home/dgx1382/students/PG/2022/amlan/download/nltk_data')
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import transformers 
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler

#read files and save a csv file 


list = {0:0, 1:1}



#train test split
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')
print(train)
print(test)
train.Result = train['Result'].map(list)
test.Result = test['Result'].map(list)

print(train.groupby(['Result']).count())

print(test.groupby(['Result']).count())

#tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

encoded_query_train = tokenizer(
    train["Query"].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512,
    return_tensors='pt',
    truncation=True
)
encoded_statute_train = tokenizer(
    train["Statute"].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512,
    return_tensors='pt',
    truncation=True
)

encoded_query_test = tokenizer(
    test["Query"].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512,
    return_tensors='pt',
    truncation=True
)
encoded_statute_test = tokenizer(
    test["Statute"].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=512,
    return_tensors='pt',
    truncation=True
)


query_input_ids_train = encoded_query_train['input_ids']
query_attention_masks_train = encoded_query_train['attention_mask']
statute_input_ids_train = encoded_statute_train['input_ids']
statute_attention_masks_train = encoded_statute_train['attention_mask']
labels_train = torch.tensor(train.Result.values)#torch.tensor(df[df.data_type=='train'].category.values)


query_input_ids_test = encoded_query_test['input_ids']
query_attention_masks_test = encoded_query_test['attention_mask']
statute_input_ids_test = encoded_statute_test['input_ids']
statute_attention_masks_test = encoded_statute_test['attention_mask']
labels_test = torch.tensor(test.Result.values)#torch.tensor(df[df.data_type=='test'].category.values)


dataset_train = TensorDataset(query_input_ids_train,query_attention_masks_train,statute_input_ids_train,statute_attention_masks_train,labels_train)
dataset_test = TensorDataset(query_input_ids_test,query_attention_masks_test,statute_input_ids_test,statute_attention_masks_test,labels_test)


def find_gpus(nums):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_free_gpus')
    with open('tmp_free_gpus', 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [ (idx,int(x.split()[2]))
                              for idx,x in enumerate(frees) ]
    idx_freeMemory_pair.sort(key=lambda my_tuple:my_tuple[1],reverse=True)
    usingGPUs = [str(idx_memory_pair[0])
                    for idx_memory_pair in idx_freeMemory_pair[:nums] ]
    usingGPUs =  ','.join(usingGPUs)
    print('using GPU idx: #', usingGPUs)
    return usingGPUs

os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(nums=4)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          ### New layers:
          self.lstm = nn.LSTM(768, 256, batch_first=True,bidirectional=True)
          self.fc = nn.Linear(512, 1)
          

    def forward(self, query_input_ids,query_attention_masks,statute_input_ids,statute_attention_masks):
          output1 = self.bert(
               query_input_ids, 
               attention_mask=query_attention_masks)

          output2 = self.bert(
               statute_input_ids,
               attention_mask=statute_attention_masks)

          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          output1 = (output1.last_hidden_state[:,0,:])#.reshape(1,-1)
          output2 = (output2.last_hidden_state[:,0,:])#.reshape(1,-1)
          out = torch.cosine_similarity(output1, output2) ### finding cosine similarity between bert embedding of query and statute
          #out = torch.mean(out, 1, True)
          out  = torch.sigmoid(out)
          out = out.unsqueeze(1)

          return out

#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = len(list),output_attentions = False,output_hidden_states = False)
#model = BertModel.from_pretrained("bert-base-uncased")
model = CustomBERTModel()

batch_size = 16

dataloader_train = DataLoader(dataset_train,sampler=RandomSampler(dataset_train),batch_size=batch_size)

dataloader_test = DataLoader(dataset_test,sampler=RandomSampler(dataset_test),batch_size=batch_size)

epochs = 10
#optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
#loss_fn = nn.BCELoss()
#def custom_BCL(seq_reduction='none'):
def custom_BCLoss(y_pred,y_true):
    #y_pred = y_pred.cpu().detach().numpy()
    #y_true = y_true.cpu().detach().numpy()
    prior_pos = torch.mean(y_true,1,keepdims=True)
    prior_neg = torch.mean(1-y_true,1,keepdims=True)
    #eps = 1e-10
    eps = 0.00001
    weight = y_true / (prior_pos + eps) + (1-y_true) / (prior_neg + eps)
    ret = -weight * (y_true * (torch.log(y_pred + eps)) + (1-y_true) * (torch.log(1-y_pred + eps)))
    seq_reduction='mean'
    #eps = 0.001
    #ret = (y_true * (tf.math.log(y_pred + eps)) + (1-y_true) * (tf.math.log(1-y_pred + eps)))
    if seq_reduction=='mean':
        return torch.mean(ret)
    elif seq_reduction=='none':
        return ret
    #return loss

optimizer = AdamW(model.parameters(),lr = 1e-5,eps = 1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps = len(dataloader_train)*epochs)
avg ='macro'

def confusion_matrix_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return str(confusion_matrix(labels_flat, preds_flat))

def precision_score_func(preds, labels,avg):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, preds_flat, average = avg)

def recall_score_func(preds, labels, avg):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, preds_flat, average = avg)

def f1_score_func(preds, labels, avg):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = avg)
 
def print_accuracy(preds, labels):
    label_dict_inverse = {v: k for k, v in list.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    num=0
    den=0

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        num+=len(y_preds[y_preds==label])
        den+=len(y_true)
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')
    
    print(f'Final accuracy percentage: {(num/den)*100}%\n')

def get_final_result(preds, labels, avg):
    print('Precision : {}\n'.format(precision_score_func(preds, labels, avg)))
    print('Recall : {}\n'.format(recall_score_func(preds, labels, avg)))
    print('F1 score : {}\n'.format(f1_score_func(preds, labels, avg)))
    print('Confusion Matrix : \n{}\n'.format(confusion_matrix_func(preds, labels)))

model.cuda()


def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader_val):
        
        batch = tuple(b.cuda() for b in batch)
        
        inputs = {'query_input_ids':      batch[0],
                  'query_attention_masks': batch[1],
                  'statute_input_ids':  batch[2],
                  'statute_attention_masks': batch[3]
                 }
        with torch.no_grad():        
            outputs = model(**inputs)

        targets = batch[4].reshape(-1,1)
        targets = targets.to(torch.float32)
        #calculate loss
        #loss = loss_fn(outputs, targets)
        loss = custom_BCLoss(outputs, targets)

        loss_val_total += loss.item()

        #backprop
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #scheduler.step()

        outputs = outputs.detach().cpu()
        logits=[]
        for x in outputs:
            logits.append([1-x, x])
        label_ids = batch[4].cpu()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs+1)):
    model.train()
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train,
                        desc='Epoch {:1d}'.format(epoch),
                        leave=False,
                        disable=False)

    for batch in progress_bar:
        model.zero_grad()
        batch = tuple(b.cuda() for b in batch)
        inputs = {'query_input_ids':      batch[0],
                  'query_attention_masks': batch[1],
                  'statute_input_ids':  batch[2],
                  'statute_attention_masks': batch[3]
                 }

        outputs = model(**inputs)
        targets = batch[4].reshape(-1,1)
        targets = targets.to(torch.float32)

        #calculate loss
        loss = custom_BCLoss(outputs, targets)
        loss_train_total +=loss.item()

        #accuracy
        #predicted = model(torch.tensor(x,dtype=torch.float32))
        #acc = (predicted.reshape(-1).detach().numpy().round() == y).mean()
        
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})

    #torch.save(model.state_dict(), f'Models/BERT_ft_Epoch{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_test)
    val_pre = precision_score_func(predictions, true_vals, avg)
    val_rec = recall_score_func(predictions, true_vals, avg)
    val_f1 = f1_score_func(predictions, true_vals, avg)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'Precision Score : {val_pre}')
    tqdm.write(f'Recall Score : {val_rec}')
    tqdm.write(f'F1 Score : {val_f1}')


print_accuracy(predictions, true_vals)
get_final_result(predictions, true_vals, avg)

print('done')
'''

