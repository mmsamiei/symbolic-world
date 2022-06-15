import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn as nn
import random
import numpy as np
import math
from tqdm.auto import tqdm
import torchmetrics
import itertools

device = torch.device('cuda')

experiment_config = {
    'obj_size': 4,
    'obs_size': 8,
}
model_config = {
    'd_model': 32,
    'dim_feedforward': 64,
    'n_head': 8,
    'num_layers': 1,
    'num_obj': 8,
}
training_config = {
    'batch_size':512,
    'lr': 5e-5,
    'max_step': 3000
}

class MySimpleModel(nn.Module):
    
    def __init__(self, model_config):
        super(MySimpleModel, self).__init__()
        d_model = model_config['d_model']
        n_heads = model_config['n_head']
        n_layers = model_config['num_layers']
        n_objects = model_config['num_obj']
        n_clss = 5
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_layer = torch.nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.token_embedding = torch.nn.Embedding(n_objects, d_model)
        self.clss_embedding = torch.nn.Embedding(n_clss, d_model)
        self.fc1 = torch.nn.Linear(d_model, 2)
        
    def forward(self, x, clss):
        '''
        x : long tensor: [batch, seq_len]
        clss: long tensor: [batch, 1]
        return: tensor: [batch, 2]
        '''
        temp = x
        temp = self.token_embedding(temp)
        clss_embedding = self.clss_embedding(clss)
        temp = torch.cat((clss_embedding, temp), dim=1)
        temp = self.transformer_layer(temp)
        temp = temp[:, 0, :]
        temp = self.fc1(temp)
        return temp


class SymbolicWorldIterableDataset(IterableDataset):
    def __init__(self, num_objects=4, observation_size=8):
        self.num_objects = num_objects
        self.observation_size = observation_size
        pass

    def generate_observation(self):
        observation = torch.randint(0, self.num_objects, (self.observation_size,))
        parity_1 = ((observation == 1).sum())%2
        return {'observation':observation, 'parity_1':parity_1}

    def __iter__(self):
        while(True):
            yield self.generate_observation()

def eval(test_loader):
    with torch.no_grad():
            test_f1s = []
            for j, batch in enumerate(test_loader):
                x = batch['observation'].to(device)
                clss = torch.ones((x.shape[0], 1)).long().to(device)
                y = batch['parity_1'].to(device)
                output = model(x, clss)
                preds = output.argmax(dim=1)
                test_f1s.append(f1_module(preds, y).item())
                if j > 10:
                    break
            f1 = sum(test_f1s)/len(test_f1s)
            return round(f1, 4)

if __name__ == '__main__':
    iterable_dataset = SymbolicWorldIterableDataset(experiment_config['obj_size'], experiment_config['obs_size'])
    test_loaders_list = []
    for observation_size in [4, 16, 64]:
        test_loader_name = f'observation_size_{observation_size}'
        test_loader_data = DataLoader(SymbolicWorldIterableDataset(observation_size=observation_size, num_objects=experiment_config['obj_size']),\
            batch_size=training_config['batch_size'])
        test_loaders_list.append({
            'name': test_loader_name,
            'data_loader': test_loader_data
        })
    train_loader = DataLoader(iterable_dataset, batch_size=training_config['batch_size'])

    n_layers_options = [2]
    n_head_options = [2,4,8]
    d_model_options = [32]
    configs = []
    for config_option in itertools.product(d_model_options, n_head_options, n_layers_options):
        new_config = {}
        new_config['model_config'] = model_config.copy()
        new_config['model_config']['d_model'] = config_option[0]
        new_config['model_config']['dim_feedforward'] = config_option[0] * 2
        new_config['model_config']['n_head'] = config_option[1] 
        new_config['model_config']['num_layers'] = config_option[2]
        configs.append(new_config)

    for config in tqdm(configs):
        print(config['model_config']['num_layers'], '\t', config['model_config']['n_head'], '\t',  new_config['model_config']['d_model'] )
        model = MySimpleModel(config['model_config']).to(device)
        loss_module = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config['lr'])
        f1_module = torchmetrics.F1Score(num_classes=2).to(device)
        loss_history = []
        max_steps = training_config['max_step']
        for i, batch in tqdm(enumerate(train_loader), total=max_steps):
            x = batch['observation'].to(device)
            clss = torch.ones((x.shape[0], 1)).long().to(device)
            y = batch['parity_1'].to(device)
            optimizer.zero_grad()
            output = model(x, clss)
            loss = loss_module(output, y)
            try:
                loss_history.append(loss.item())
            except:
                print(loss.item())
            loss.backward()
            optimizer.step()
            if i > max_steps:
                print("loss: {:.2e}".format(sum(loss_history)/len(loss_history)))
                loss_history = []
                train_f1 = f1_module(output.argmax(dim=1), y).item()
                print('train f1: ', round(train_f1,4))
                string = []
                for test_loader in test_loaders_list:
                    test_acc = eval(test_loader['data_loader'])
                    string.append(str(test_acc))
                print('\t'.join(string))
                print('------------------------')
                break

