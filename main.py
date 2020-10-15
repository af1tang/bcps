#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anonymous
"""

import os
import numpy as np
import argparse
import pickle
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim

from models import *
from utils import *

from sklearn.model_selection import *
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_curve, auc as auc_score, confusion_matrix, f1_score
from sklearn.utils import shuffle


#### FS and PL Adv. Training #####
def training_loop(X,y, opts):
    """The incremental training loop.
        * Saves checkpoints of D and G every opts.checkpoint_every iterations.
        * Saves generated samples every opts.sample_every iterations.
    """
    # Note that this X and y are actually X_tr and y_tr, from a single split scheme.
    # Set Budget.
    opts.total_budget = int(opts.budget * X.shape[0]* X.shape[1])
    
    # Create FS (G) and PL (D)
    G, D = create_model(opts)

    # Create optimizers for FS (G) and PL (D)
    g_optimizer = optim.Adam(G.parameters(), opts.learning_rate)
    d_optimizer = optim.Adam(D.parameters(), opts.learning_rate)
    
    iteration = 1

    for epoch in range(opts.num_epochs):
        # balanced subsampling.
        xs, ys = balanced_subsample(X,y)
        total_train_iters = round(opts.num_epochs * len(xs) / opts.batch_size)
        dataset = list(zip(xs,ys))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                             shuffle=True)
        # initialize regularizer parameter
        beta = opts.l1_reg
        #alpha = 0.001

        for batch in dataloader:
            real_x, labels = batch
            # make torch variables.
            real_x, labels = to_var(real_x), to_var(labels).long().squeeze()
            batch_size, seq_len, _ = real_x.size()

            #####################################
            ########  PL Training    #######
            #####################################

            for p in D.parameters():
                p.requires_grad = True  
            for p in G.parameters():
                p.requires_grad = False

            d_optimizer.zero_grad()

            # Criterion to use, fake and real labels.
            t_criterion = nn.CrossEntropyLoss()
            
            if epoch <= 30:
                t_loss = t_criterion(D(real_x), labels)
                t_loss.backward()
                d_optimizer.step()
                
                # Print the log info
                if iteration % opts.log_step == 0:
                    print('Iteration [{:4d}/{:4d}] | t_loss: {:6.4f} '.format(
                           iteration, total_train_iters, t_loss.data[0] ))
    
                # Save the model parameters
                if iteration % opts.checkpoint_every == 0:
                    checkpoint(iteration, G, D, opts)
                
            else:
                
                # 1. Generate sensed X (fake_x)
                mask = G(real_x)
                mask = to_data(mask.detach())
                mask[mask>.2] = 1.
                mask[mask<=.2] = 0.
                mask = to_var(torch.from_numpy(mask))
                #fake_x = real_x - mask * real_x
                fake_x = mask * real_x
                # 2. Updated PL
                t_loss = t_criterion(D(fake_x.detach()), labels) #+ t_criterion(D(real_x), labels)
                t_loss.backward()
                d_optimizer.step()
                
                #####################################
                ########   FS Training   #######
                #####################################
                s_criterion = nn.MarginRankingLoss()

                # max-margin labels: y = {-1, +1}.
                margin_labels = to_var(torch.tensor([(lambda x: 1 if x ==1 else -1)(yy) for yy in labels]))

                for p in D.parameters():  
                    p.requires_grad = False
                for p in G.parameters():
                    p.requires_grad = True
                    
                g_optimizer.zero_grad()
    
                # 1. Generate masked X from original X
                mask = G(real_x)
                fake_x = real_x - mask * real_x
    
                # 2. Compute the generator loss 
                g_loss = s_criterion(D(fake_x)[:, 0], D(fake_x)[:,1], margin_labels) + \
                beta * torch.norm(mask * real_x, p=1) 
                
                g_loss.backward()
                g_optimizer.step()
                
                # 3. Update budget constraints
                cost= to_data(mask.detach())
                cost[cost>.2] = 1.
                cost[cost<=.2] = 0.
                #cost = 1 - cost
                #cost = torch.sum(mask.detach()) - opts.total_budget
                #lambda_reg = torch.clamp(lambda_reg + alpha * cost, 0.001, 1.)
                
                # Print the log info
                if iteration % opts.log_step == 0:
                    print('Iteration [{:4d}/{:4d}] | t_loss: {:6.4f} | s_loss: {:6.4f} | total_cost: {:6.0f}'.format(
                           iteration, total_train_iters, t_loss.data[0], g_loss.data[0], np.sum(cost) ))

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                checkpoint(iteration, G, D, opts)

            iteration += 1
            
            for p in D.parameters():  
                p.requires_grad = False
            for p in G.parameters():
                p.requires_grad = False
                
    return G, D

def create_model(opts):
    """Builds the generators and discriminators.
    """
    # FS is the "Generator"
    # PL is the "Discriminator"
    G = Generator(feature_size = opts.feature_size,hidden_size = opts.g_hidden_size)
    D = Discriminator(feature_size=opts.feature_size, hidden_size = opts.d_hidden_size,
                  output_size = opts.logits_size)

    if torch.cuda.is_available():
        G.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G, D

#### Inference #####
def inference_loop(X, y, G,D, opts):
    '''
    y: labels
    X: inputs
    G: dynamic sensor from FS
    '''
    #aux_size = len(c[-1])
    # 1. Make dataloader for X,y.
    #dataset = list(zip(X, y))
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                             #shuffle=True)
    # 2. Create Models
    mlp_r = MLP(feature_size = opts.feature_size*3, hidden_size = opts.hidden_size, output_size = opts.output_size)
    gru_r = GRU(feature_size = opts.feature_size, hidden_size = opts.hidden_size, output_size = opts.output_size)
    lr_r = LR(penalty = 'l1', C=1e5, warm_start = True, max_iter = 1000)
    
    mlp_f = MLP(feature_size = opts.feature_size*3, hidden_size = opts.hidden_size, output_size = opts.output_size)
    gru_f = GRU(feature_size = opts.feature_size, hidden_size = opts.hidden_size, output_size = opts.output_size)
    lr_f = LR(penalty = 'l1', C=1e5, warm_start = True, max_iter = 1000)
    
    attn_r = SelfAttention(feature_size = opts.feature_size, hidden_size = opts.hidden_size, output_size = opts.output_size)
    attn_f = AttentionGRU(feature_size = opts.feature_size, hidden_size = opts.hidden_size, output_size = opts.output_size)
    
    # 3. initialize optimizers for deep models
    mr_optimizer = optim.Adam(mlp_r.parameters(), opts.learning_rate)
    gr_optimizer = optim.Adam(gru_r.parameters(), opts.learning_rate)
    mf_optimizer = optim.Adam(mlp_f.parameters(), opts.learning_rate)
    gf_optimizer = optim.Adam(gru_f.parameters(), opts.learning_rate)
    ar_optimizer = optim.Adam(attn_r.parameters(), opts.learning_rate)
    af_optimizer = optim.Adam(attn_f.parameters(), opts.learning_rate)
    d_optimizer = optim.Adam(D.parameters(), opts.learning_rate)
    #3. training loop
    iteration = 1

    for epoch in range(opts.train_epochs):
        xs, ys = balanced_subsample(X,y)
        total_train_iters = round(opts.train_epochs * len(xs) / opts.batch_size)
        dataset = list(zip(xs, ys))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                             shuffle=True)
        for batch in dataloader:
            real_x, labels = batch
            real_x, labels = to_var(real_x), to_var(labels).squeeze()
            batch_size, seq_len, _ = real_x.size()
            mask = G(real_x)
            mask = to_data(mask.detach())
            mask[mask>.2] = 1.
            mask[mask<=.2] = 0.
            mask = to_var(torch.from_numpy(mask))
            
            #fake_x = real_x - mask * real_x
            fake_x = mask * real_x
            avg_real_x = collapsed_features(real_x) #torch.cat([real_x.min(dim=1)[0], real_x.max(dim=1)[0], real_x.mean(dim=1)], dim = 1)
            avg_fake_x = collapsed_features(fake_x) #torch.cat([fake_x.min(dim=1)[0], fake_x.max(dim=1)[0], fake_x.mean(dim=1)], dim = 1)
            labels = labels.reshape(batch_size, 1)
            
            # Criterion to use, BCE obviously.
            criterion = nn.BCELoss()
            d_criterion = nn.CrossEntropyLoss()
            
            # Grad Steps.
            lambda_1 = opts.l1_reg
            # mlp_r backprop
            mr_optimizer.zero_grad()
            mlp_r_loss = criterion(mlp_r(avg_fake_x.data), labels)
            mlp_r_loss.backward()
            mr_optimizer.step()
            # mlp_f backprop
            mf_optimizer.zero_grad()
            mlp_f_loss = criterion(mlp_f(avg_real_x.data), labels)
            mlp_f_loss.backward()
            mf_optimizer.step()
            # gru_r backprop
            gr_optimizer.zero_grad()
            gru_r_loss = criterion(gru_r(real_x.data), labels)
            gru_r_params = torch.cat([x.view(-1) for x in gru_r.parameters()])
            gru_r_loss += lambda_1 * torch.norm(gru_r_params, 1)
            gru_r_loss.backward()
            gr_optimizer.step()
            # gru_f backprop
            gf_optimizer.zero_grad()
            gru_f_loss = criterion(gru_f(fake_x.data), labels)
            gru_f_loss.backward()
            gf_optimizer.step()
            #attn_r backprop
            ar_optimizer.zero_grad()
            ar_loss = criterion(attn_r(real_x.data)[0], labels)
            ar_loss.backward()
            ar_optimizer.step()
            #attn_f backprop
            af_optimizer.zero_grad()
            af_loss = criterion(attn_f(fake_x.data)[0], labels)
            af_loss.backward()
            af_optimizer.step()
            #d backprop
            for p in D.parameters():  
                p.requires_grad = True
            d_optimizer.zero_grad()
            d_loss = d_criterion(D(fake_x.data), labels.long().reshape(batch_size,))
            d_loss.backward()
            d_optimizer.step()
            
            # Reporting
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | mlp_loss: {:6.4f} | gru_loss: {:6.4f} | attn_loss: {:6.4f}'.format(
                       iteration, total_train_iters, mlp_f_loss.data[0], gru_r_loss.data[0], attn_loss.data[0]))
            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                models = {'mlp_r': mlp_r, 'mlp_f': mlp_f, 'D': D,
                          'gru_r': gru_r, 'gru_f':gru_f, 
                          'attn_r': attn_r, 'attn_f': attn_f}
                inference_checkpoint(iteration,models, opts)
            iteration += 1
    var_x = to_var(torch.from_numpy(X))
    mask = G(var_x)
    mask = to_data(mask.detach())
    mask[mask>.2] = 1.
    mask[mask<=.2] = 0.
    mask = to_var(torch.from_numpy(mask))
    generated_x = mask * var_x

    avg_x = collapsed_features(var_x)
    avg_g = collapsed_features(generated_x)
    avg_x, avg_g = to_data(avg_x), to_data(avg_g)
    lr_r = lr_r.fit(avg_x, y)
    lr_f = lr_f.fit(avg_g, y)
    
    models = {'mlp_r': mlp_r, 'mlp_f': mlp_f, 
              'gru_r': gru_r, 'gru_f':gru_f, 'D': D,
              'lr_r': lr_r, 'lr_f': lr_f, 
              'attn_r': attn_r, 'attn_f': attn_f}
    return models

#### Evaluation ####
def evaluate(X, y, G, models, opts):
    print()
    print("Evaluating Models ...")
    print()
    data = {}
    real_x = to_var(torch.from_numpy(X))
    mask = G(real_x)
    mask = to_data(mask.detach())
    mask[mask>.2] = 1.
    mask[mask<=.2] = 0.
    mask = to_var(torch.from_numpy(mask))

    fake_x = mask * real_x
    avg_x = collapsed_features(real_x)
    avg_g = collapsed_features(fake_x)
    for name, model in models.items():
        if 'lr_r' == name:
            yhat= model.predict(to_data(avg_x))
        elif 'lr_f' == name:
            yhat = model.predict(to_data(avg_g))
        elif 'mlp_r' == name:
            yhat = model(avg_x)
            yhat = to_data(yhat)
        elif 'mlp_f' == name:
            yhat = model(avg_g)
            yhat = to_data(yhat)
        elif 'attn_r' == name:
            yhat = model(real_x)
            yhat = to_data(yhat[0])
        elif 'attn_f' == name:
            yhat = model(fake_x)
            yhat = to_data(yhat[0])  
        elif'D' == name:
            yhat = model(fake_x)
            yhat = to_data(yhat[:,-1])
        elif 'gru_r' == name:
            yhat = model(real_x)
            yhat = to_data(yhat)
        elif 'gru_f' == name:
            yhat = model(fake_x)
            yhat = to_data(yhat)
        roc_auc, f1, sen, spec = scoring(y, yhat)
        data[name] = {'auc': roc_auc, 'f1': f1, 'sen': sen, 'spec': spec}
    return data

def scoring(y_te, yhat):
    fpr, tpr, thresholds = roc_curve(y_te, yhat)
    roc_auc = auc_score(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    yhat[yhat>=optimal_threshold]=1; yhat[yhat<optimal_threshold]=0
    yhat=[int(i) for i in yhat]
    #matrix = confusion_matrix(y_te, yhat)
    tn, fp, fn, tp = confusion_matrix(y_te, yhat).ravel()
    sen=1.0* (tp/(tp+fn))
    spec=1.0* (tn/(tn+fp))
    f1=f1_score(y_te,yhat)
    return roc_auc, f1, sen, spec

### Beta Decay Test ###
def beta_decay(X_tr, y_tr, 
               X_te, y_te,
               opts, betas = [  5e-5, 2.5e-5, 7.5e-5, 1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5]):
    beta_data = {}
    for beta in betas:
        opts.l1_reg = beta
        G,D = training_loop(X_tr,y_tr,opts)
        mask = G(to_var(torch.from_numpy(X_te)))
        cost = to_data(mask.detach())
        cost[cost>.2]=1.; cost[cost<=.2] = 0.
        budget = np.sum(cost) / (48*19*len(X_te))
        beta_data[beta] = {'data': evaluate(X_te, y_te, G, {'D': D}, opts), 'budget': budget,
                 'G': G.state_dict(), 'D': D.state_dict()} 
    return beta_data
    
def syn_exp(opts, 
            filename= '~/tmp/synthetic.data'):
    #### Read files ####
    import random
    def read_data(filename):
        with open(filename, 'r') as f:
            data = f.readlines()
        data = np.array([np.array([float (dat) for dat in d.split()]) for d in data])
        return data

    def generate_window_data(data):
        windows = np.array(list(window(data[:,1], 1000)))
        return windows[random.sample(range(len(windows)),1000)]
    data = read_data(filename)
    windows = generate_window_data(data)
    windows = windows.reshape(1000,100,10)
    
    #add noise
    X = np.random.rand(windows.shape[0], windows.shape[1], windows.shape[2]) * -10
    X[:, 10:30, 2:5] = windows[:, 10:30, 2:5]
    X[:, 90:, 1:3] = windows[:, 90:, 1:3]
    X[:, 80:, 8:] = windows[:, 80:, 8:]
    #ground_truth
    ground_truth = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    ground_truth[:, 10:30, 2:5] = 1
    ground_truth[:, 90:, 1:3] =1
    ground_truth[:, 80:, 8:] =1

    y = []    
    for i in range(len(X)):
        if (np.sum(X[i][10:30, 2:5]) + np.sum(X[i][90:, 1:3]) + np.sum(X[i][80:, 8:]) >0):
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    
    data = np.array(list(zip(X,y)))
    skf = StratifiedKFold(n_splits=5, random_state=13 )
    train_index, test_index = list(skf.split(X, y))[0]
    X_tr, X_te = X[train_index], X[test_index]
    y_tr, y_te = y[train_index], y[test_index]
    stats = {}
    
    #change opts
    opts.g_hidden_size = 32
    opts.d_hidden_size = 32
    opts.feature_size = 10
    opts.l1_reg=5e-6
    
    #BCPS evaluation
    G,D = training_loop(X_tr,y_tr,opts)
    mask = G(to_var(torch.from_numpy(X_te)))
    cost = to_data(mask.detach())
    cost[cost>.2]=1.; cost[cost<=.2] = 0.
    budget = np.sum(cost) / (48*19*len(X_te))
    
    #attention and feature selection evaluation
    gru = GRU(feature_size = opts.feature_size, hidden_size = opts.hidden_size, output_size = opts.output_size)    
    
    attn= SelfAttention(feature_size = opts.feature_size, hidden_size = opts.hidden_size, output_size = opts.output_size)
    
    # 3. initialize optimizers for deep models
    g_optimizer = optim.Adam(gru.parameters(), opts.learning_rate)
    a_optimizer = optim.Adam(attn.parameters(), opts.learning_rate)

    #3. training loop
    iteration = 1

    for epoch in range(opts.train_epochs):
        total_train_iters = round(opts.train_epochs * len(X_tr) / opts.batch_size)
        dataset = list(zip(X_tr, y_tr))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                             shuffle=True)
        for batch in dataloader:
            xs, ys = batch
            xs, ys = to_var(xs), to_var(ys).squeeze()
            batch_size, seq_len, _ = xs.size()
            
            # Criterion to use, BCE obviously.
            criterion = nn.BCELoss()
            
            # Grad Steps.
            lambda_1 = 1e-5
            # gru_r backprop
            g_optimizer.zero_grad()
            gru_loss = criterion(gru(xs.data), ys)
            gru_params = torch.cat([x.view(-1) for x in gru.parameters()])
            gru_loss += lambda_1 * torch.norm(gru_params, 1)
            gru_loss.backward()
            g_optimizer.step()

            #attn_r backprop
            a_optimizer.zero_grad()
            a_loss = criterion(attn(xs.data)[0], ys)
            a_loss.backward()
            a_optimizer.step()
            
            # Reporting
            if iteration % opts.log_step == 0:
                print('Iteration [{:4d}/{:4d}] | gru_loss: {:6.4f} | attn_loss: {:6.4f}'.format(
                       iteration, total_train_iters, gru_loss.data[0], a_loss.data[0]))

            iteration += 1
    
    stats= {'data': evaluate(X_te, y_te, G, {'D': D, 'gru_r': gru, 'attn_r': attn}, opts), 'budget': budget,
             'G': G.state_dict(), 'D': D.state_dict()} 

    return data, stats, mask, ground_truth

### Main ###
def main(opts):
    X = np.load(opts.features_dir)
    y = get_task(opts)
    stats = {}; fold = 0
    skf = StratifiedKFold(n_splits=5 )
    stats[1] = {}
    ### Main Train and Testing Loops ###
    #=============================================================================
    for train_index, test_index in skf.split(X, y):
        fold +=1
        print ("KFold #{0}".format(fold))
        X_tr, X_te = X[train_index], X[test_index]
        y_tr, y_te = y[train_index], y[test_index]
        G, D = training_loop(X_tr,y_tr,opts)
        models = inference_loop(X_tr, y_tr, G, D, opts)
        data = evaluate(X_te, y_te, G, models, opts)
        with open(opts.checkpoint_path + '/data_' + str(fold), 'wb') as f:
            pickle.dump(data, f)
        stats[fold] = data
        print()
        print(data)
        print()

    ### Sparsity Trade-Off Study ###    
    beta_data = beta_decay(X_tr, y_tr, X_te, y_te, opts)
    report = reporting(stats)
    with open(opts.checkpoint_path + '/report' + str(fold), 'wb') as f:
            pickle.dump(report, f)
    return G, D, models, stats, skf, beta_data

def get_task(opts):
    with open(opts.y_dir, 'rb') as f:
        labels = pickle.load(f)
    dct = {'mort':0, 'sep': 1}
    #conditionals = [ list(yy[0]) + [ yy[dct[opts.task]] ] for yy in labels] 
    task = [yy[dct[opts.task]] for yy in labels]
    opts.output_size = 1
    return  np.array(task)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Training hyper-parameters
    parser.add_argument('--features_dir', type=str, default='~/tmp/X.npy',
                        help='Path to the timeseries features (X).')
    parser.add_argument('--y_dir', type=str, default='~/tmp/y',
                        help='Path to the task labels (Y) file.')
    

    parser.add_argument('--task', choices = ['mort', 'sep' ], default='mort',
                        help='Target task: mortality (mort), sepsis(sep).')
    parser.add_argument('--budget', type = float, default = .5e-5,
                        help='Budget (%) for observations.')

    parser.add_argument('--feature_size', type=int, default=19,
                        help='The size of feature space.')
    parser.add_argument('--logits_size', type=int, default=2,
                        help='The size of output space.')
    parser.add_argument('--output_size', type=int, default=1,
                        help='The size of output space.')
    parser.add_argument('--l1_reg', type=float, default=5e-6,
                        help='Lasso penalty for baseline inference model.')
    parser.add_argument('--g_hidden_size', type=int, default=128,
                        help='The size of S hidden units (for deep models).')
    parser.add_argument('--d_hidden_size', type=int, default=128,
                        help='The size of T hidden units (for deep models).')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='The size of inference hidden units (for deep models).')
    parser.add_argument('--num_epochs', type=int, default=45,
                        help='The max number of epochs for FS and PL.')
    parser.add_argument('--train_epochs', type=int, default=30,
                        help='The max number of epochs for inference models.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The number of examples in a batch.')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='The learning rate (default 0.005)')

    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--checkpoint_every', type=int , default=200)
    parser.add_argument('--checkpoint_dir', type=str, default='~/tmp/model',
                        help='Set the directry to store the best model checkpoints.')

    return parser


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
        else:
            print('{:>30}: {:<30}'.format(key, 'None').center(80))
    print('=' * 80)


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    print_opts(opts)

    model_name = '{}-{}'.format(opts.task, opts.budget)
    opts.checkpoint_path = os.path.join(opts.checkpoint_dir, model_name)

    create_dir_if_not_exists(opts.checkpoint_path)
    create_dir_if_not_exists(os.path.join(opts.checkpoint_path, 'scores'))

    scores_folder = os.path.join(opts.checkpoint_path, 'scores')
    G, D, models, stats, skf, beta_data = main(opts)

    with open(scores_folder + '/raw_stats', 'wb') as f:
        pickle.dump(stats, f)
    print ("Done!")
    
        