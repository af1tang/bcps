#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: anonymous
"""
import torch
import torch.nn.functional as F
import progressbar
import pandas as pd
import gensim
import os
import numpy as np
from itertools import tee, islice
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def one_hot(arr, size):
    onehot = np.zeros((len(arr),size), dtype = int)
    for i in range(len(arr)):
        if not np.isnan(arr[i]):            
            onehot[i, int(arr[i])]=1
    return onehot

def bow_to_ohv(dct):
    '''converts bag of words to one-hot format.
    dct: {hadm_id: (X, y)}
    '''
    xy = {}
    for h in dct.keys():
        X, y = [], dct[h][1]
        for t in dct[h][0]:
            X.append(np.array( [(lambda x: 1 if xx > 0 else 0)(xx) for xx in t] ) )
        X = np.array(X)
        xy[h] = (X,y)
    return xy

def bow_sampler(x, size):
    if not pd.isnull(x).all():
        bow = np.sum(one_hot(x, size), axis=0) 
        bow = np.array([(lambda x: 1 if x >0 else 0)(xx) for xx in bow])
        first = one_hot(x,size)[0]
        last = one_hot(x,size)[-1]
        return [first, bow, last]
    else:
        return np.nan

def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
    
### Progressbar tools ### 
def make_widget():
    widgets = [progressbar.Percentage(), ' ', progressbar.SimpleProgress(), ' ', 
                                 progressbar.Bar(left = '[', right = ']'), ' ', progressbar.ETA(), ' ', 
                                 progressbar.DynamicMessage('LOSS'), ' ',  progressbar.DynamicMessage('PREC'), ' ',
                                 progressbar.DynamicMessage('REC')]
    bar = progressbar.ProgressBar(widgets = widgets)
    return bar


### Find nearest timestamps ###
def find_prev(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx == 0:
        return array[idx]
    else:
        return array[idx-1]

def find_next(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx == len(array) -1:
        return array[idx]
    else:
        return array[idx+1]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


### utility function ###

def flatten(lst):
    make_flat = lambda l: [item for sublist in l for item in sublist]
    return make_flat(lst)    

def balanced_subsample(x,y,subsample_size=1.0):
    class_xs = []
    min_elems = None
    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]
    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)
    xs = []
    ys = []
    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

def hierarchical_subsample(x, dx, y,subsample_size=1.0):
    from sklearn.utils import shuffle
    class_xs = []
    class_dxs = []
    min_elems_x = None
    min_elems_d = None

    for yi in np.unique(y):
        elems_x = x[(y == yi)]
        elems_d = dx[(y==yi)]
        class_xs.append((yi, elems_x))
        class_dxs.append((yi, elems_d))
        if min_elems_x == None or elems_x.shape[0] < min_elems_x:
            min_elems_x = elems_x.shape[0]
            min_elems_d = elems_d.shape[0]

    use_elems_x = min_elems_x
    use_elems_d = min_elems_d
    if subsample_size < 1:
        use_elems_x = int(min_elems_x*subsample_size)
        use_elems_d = int(min_elems_d*subsample_size)

    xs = []
    dxs = []
    ys = []

    for lst1, lst2 in zip(class_xs, class_dxs):
        ci = lst1[0]
        this_xs = lst1[1]
        this_dxs = lst2[1]
        
        if len(this_xs) > use_elems_x:
            this_xs, this_dxs = shuffle(this_xs, this_dxs)

        x_ = this_xs[:use_elems_x]
        d_ = this_dxs[:use_elems_d]
        y_ = np.empty(use_elems_x)
        y_.fill(ci)

        xs.append(x_)
        dxs.append(d_)
        ys.append(y_)

    xs = np.concatenate(xs)
    dxs = np.concatenate(dxs)
    ys = np.concatenate(ys)

    return xs, dxs, ys

#### Pickling Tools ####
def checkpoint(iteration, G, D, opts):
    """Saves the parameters of the generator G and discriminator D.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G.pkl')
    D_path = os.path.join(opts.checkpoint_dir, 'D.pkl')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)

def inference_checkpoint(iteration, models, opts):
    '''Saves parameters of inference models.
    '''
    for name, model in models.items():
        path = os.path.join(opts.checkpoint_dir, '{}.pkl'.format(name))
        torch.save(model.state_dict(), path)
    
def large_save(dct, file_path):
    '''dct: {k: v}
    '''
    lst = sorted(dct.keys())
    chunksize =10000
    #chunk bytes
    bytes_out= bytearray(0)
    for idx in range(0, len(lst), chunksize):
        bytes_out += pickle.dumps(dict([(k,v) for k,v in dct.items() if k in lst[idx: idx+ chunksize]]))
    with open(file_path, 'wb') as f_out:
            for idx in range(0, len(bytes_out), chunksize):
                f_out.write(bytes_out[idx:idx+chunksize])
    #split files
    for idx in range(0, len(lst), chunksize):
        chunk = dict([(k,v) for k,v in dct.items() if k in lst[idx: idx+ chunksize]])
        with open(file_path+'features_'+str(idx+chunksize), 'wb') as f_out:
            pickle.dump(chunk, f_out, protocol=2)

def large_read(file_path):
    import os.path
    bytes_in = bytearray(0)
    max_bytes = int(1e5)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data

def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
#### Feature Conversions ####
def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    x = x.float()
    return torch.autograd.Variable(x)


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def collapsed_features(X):
    avg_x = torch.cat([X.min(dim=1)[0], X.max(dim=1)[0], X.mean(dim=1)], dim = 1)
    return avg_x

#### Reporting Functions ####
def reporting(stats):
    keys = list(stats[1].keys())
    report = {}
    for key in keys:
        tmp = np.array([[stats[num][key]['auc'],stats[num][key]['f1'], 
                         stats[num][key]['sen'], stats[num][key]['spec']] for num in stats.keys()])
        dct = {'auc': (np.mean(tmp[:,0]),np.std(tmp[:,0])), 'f1': (np.mean(tmp[:,1]),np.std(tmp[:,1])),
            'sen': (np.mean(tmp[:,2]),np.std(tmp[:,2])), 'spec': (np.mean(tmp[:,3]),np.std(tmp[:,3])) }
        
        report[key] = dct
    return report

def visualize_attention(X, y, attention_model, opts, save='heatmap_activations.pdf'):
    """Generates a heatmap to show where attention is focused in each decoder step.
    """
    n_samples, n_seqlen, n_features = X.shape
    activations = attention_model(to_var(torch.from_numpy(X)))[1]
    #attention_weights_matrix = np.stack(activations)
    activations = np.array([to_data(aa) for aa in activations]).reshape(n_samples, n_seqlen, n_features)
    
    ax=plt.axes()
    sns.heatmap(activations[0], cmap = sns.light_palette((220, 90, 60), input="husl"))
    ax.set_title('Sensed features over 48hr time period.')
    ax.set(ylabel='Time-Steps (Hr)', xlabel='Obs. Features')
    
    plt.show()
    
def visualize_mask(opts, mask, save = 'heatmap_mask.pdf',
                   title = "Sensed features across time.",
                   xaxis = "Obs. Features", yaxis = "Time-steps (Hr)"):
    ax=plt.axes()
    sns.heatmap(mask, cmap = sns.light_palette((220, 90, 60), input="husl"))
    ax.set_title(title)
    ax.set(ylabel=yaxis, xlabel=xaxis)
    
    plt.show()

def beta_decay_vis(beta_data, X, G):
    data = dict([(dat[0], {'auc': dat[1]['data']['D']['auc'], 'budget': dat[1]['budget'], 
                  'f1': dat[1]['data']['D']['f1'], 'sen': dat[1]['data']['D']['sen'],
                  'spec': dat[1]['data']['D']['spec']})
    for dat in beta_data.items()])
    betas= sorted(list(data.keys()))
    aucs = [data[b]['auc'] for b in betas]
    f1s = [data[b]['f1'] for b in betas]

    sns.set(style="darkgrid")
    ax = plt.axes()
    df = pd.DataFrame({'auc': aucs, 'f1': f1s, 'betas': betas})
    sns.tsplot(df,time='beta', condition = 'metric', value = 'performance')
    ax.set_title("Performance Trade-off between budget constraint vs. AUC and F1-Score")
    ax.set(ylabel="Performance", xlabel = "Budget Constraint (Beta)")
    plt.show()
    
    for beta in betas:
        G.load_state_dict(beta_data[beta]['G'])
        mask = G(to_var(torch.from_numpy(X)))
        cost = to_data(mask.detach())
        cost[cost>.2]=1.; cost[cost<=.2] = 0.
        ax=plt.axes()
        sns.heatmap(cost[0], cmap = "Blues")
        ax.set_title('BCPS over 48hr time period w/ {} budget constraint.'.format(beta))
        ax.set(ylabel='Time-Steps (Hr)', xlabel='Obs. Features')
        
        plt.show()
    return data