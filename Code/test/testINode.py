'''
@Author: Xin Han
@Date: 2020-06-07 11:24:57
LastEditTime: 2021-05-13 17:33:03
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@file_path: \StreamHC\Code\testINode.py
'''
# coding: utf-8

import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from Code.models.INode import INode
from Code.utils.anne_no_pool import add_nne_data, addNNE
from Code.utils.dendrogram_purity import dendrogram_purity, expected_dendrogram_purity
from Code.utils.file_utils import load_data, mkdir_p_safe, remove_dirs
from Code.utils.Graphviz import Graphviz

def create_i_tree(dataset, n, psi, t, rate):
    """Create trees over the same points.

    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.

    Args:
        dataset - a list of points with which to build the tree.
        n - numuber of point to intitial ik metrix
        psi - parameter of ik
        t - parameter of ik
        w - windows size
        l - ik metrix update length

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """

    met = np.array( [pt[0] for pt in dataset[:n]])
    nne_st = time.time()
    oneHot, subIndexSet, data = add_nne_data(dataset, n, psi, t)
    ind =  list(set.union(*map(set,subIndexSet)))
    indData = met[ind]
    nne_ed = time.time()
    print(nne_ed - nne_st)
    root = INode(exact_dist_thres=10)
    run_time = []
    tree_st_time = time.time()

    # history = []
    for i, pt in enumerate(data):
        if len(pt) == 3:
            #st = time.time()
            ikv = addNNE(ind,indData, pt[0], oneHot, subIndexSet)
            #et = time.time()
            #print("add time:%s"%(et-st))
            pt.append(ikv)
        if (i % 5000 == 0): 
            tree_mi_time = time.time()
            run_time.append((i, tree_mi_time - tree_st_time))

        root = root.insert(pt, delete_node=True,
                           L=5000, t=300, rate=rate)
    return root,run_time


def save_data(args, exp_dir_base, file_name):
    file_path = os.path.join(exp_dir_base, file_name+'.tsv')
    mkdir_p_safe(exp_dir_base)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (
                'dataset',
                'algorithm',
                'purity',
                'clustering_time_elapsed',
                "max_psi",
                "max_rt"))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%f\t%f\t%s\t%s\n' % (
            args['dataset'],
            args['algorithm'],
            args['purity'],
            args['clustering_time_elapsed'],
            args["max_psi"],
            args["max_rt"]))

def save_all_data(args, exp_dir_base,filename):
    file_path = os.path.join(exp_dir_base, 'allResult.tsv')
    mkdir_p_safe(exp_dir_base)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\n' % (
                'algorithm',
                'dataset',
                'purity',
                ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%f\n' % (
            args['algorithm'],
            filename,
            args['purity'],
        ))

def save_grid_search_data(args, exp_dir_base,filename):
    file_path = os.path.join(exp_dir_base, 'gridSearch.tsv')
    mkdir_p_safe(exp_dir_base)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as fout:
            fout.write('%s\t%s\t%s\t%s\t%s\n' % (
                'algorithm',
                'dataset',
				'psi',
				'beta',
                'purity',
                ))
    with open(file_path, 'a') as fout:
        fout.write('%s\t%s\t%d\t%f\t%f\n' % (
            args['algorithm'],
            filename,
			args["psi"],
			args["beta"],
            args['purity'],
        ))

def grid_search_inode(dataset, psi, t, n, rates, file_name, exp_dir_base, shuffle_index):
    ti = 0
    purity = 0
    max_ps = psi[0]
    max_rt = rates[0]
    for ps in psi:
        for rt in rates:
            print(ps,rt)
            data = deepcopy(dataset)
            sts = time.time()
            try:
                root,runTime = create_i_tree(
                  data, n, ps, t, rate=rt)
            except:
                continue
            ets = time.time()
            print("time of build tree: %s" % (ets-sts))
            ti += ets-sts
            with open("InodeTime.tsv","a") as f:
                for item in runTime:
                        f.write("%s\t%s\n" %(
                            item[0],
                            item[1]
                        ))
            #print(runTime) 
            #serliaze_tree_to_file_with_point_ids(root, "seriesTree.tsv")
            #Graphviz.write_tree("tree.dot",root)
            #print(root.point_counter)
            #print(runTime)

            pu_sts = time.time()
            dendrogram_purity = expected_dendrogram_purity(root)
            pu_ets = time.time()
            print("purity time: %s" %(pu_ets - pu_sts))
            #dendrogram_purity = 0
            print("dendrogram_purity: %s" % (dendrogram_purity))
            args = {
				'dataset': file_name+"_"+"shuffle"+"_"+str(shuffle_index),
            	'algorithm': "IKSHC",
            	'purity': dendrogram_purity,
				"psi": ps,
            	"beta": rt}
            save_grid_search_data(args,exp_dir_base=exp_dir_base, filename=file_name)
            #pu_ets = time.time()
            #print("time of calcute purity: %s" % (pu_ets-pu_sts))
            if dendrogram_purity > purity:
                max_ps = ps
                max_rt = rt
                purity = dendrogram_purity
            del data
    tim = ti/(len(psi)*len(rates))
    args = {'dataset': file_name+"_"+"shuffle"+"_"+str(shuffle_index),
            'algorithm': "IKSHC",
            'purity': purity,
            'clustering_time_elapsed': tim,
            "max_psi": max_ps,
            "max_rt": max_rt}
    save_data(args, exp_dir_base, file_name)
    save_all_data(args=args, exp_dir_base=exp_dir_base, filename = file_name)


if __name__ == "__main__":
    # def load_df(df):
    #     for item in df.values:
    #         yield([item[:-2], item[-2], item[-1]])

    # dimensions = [10]
    # size = 10
    # num_clus = 3
    # for dim in dimensions:
    #   print("TESTING DIMENSIONS == %d" % dim)
    #   dataset = create_dataset(dim, size, num_clusters=num_clus)

    # dataset = pd.DataFrame(dataset)
    # scaler = MinMaxScaler()
    # dataset.iloc[:,:-2] = scaler.fit_transform(dataset.iloc[:,:-2])
    # dataset = list(load_df(dataset))
    psi = [64, 15, 17, 21, 25]
    #psi = [15] 
    #rates = [0.8]
    rates = [0.4,0.5,0.6, 0.7, 0.8,0.9]
    t = 300
    remove = False
    file_name = "aloi"
    exp_dir_base_inode = './Code/data/aloi/'
    dati = time.strftime("%Y%m%d%H%M%S", time.localtime())
    exp_dir_base_inode = exp_dir_base_inode+dati
    shuffle_times = 10
    dataset = list(load_data("./Code/data/aloi/"+file_name+".tsv"))[:5000]
    n = int(len(dataset)/4)
    print(n)

    if remove:
        remove_dirs(file_name=file_name, exp_dir_base=exp_dir_base_inode)
    for i in range(shuffle_times):
        np.random.shuffle(dataset)
        grid_search_inode(dataset=dataset, n=n, t=t, psi=psi, rates=rates,
                          file_name=file_name, exp_dir_base=exp_dir_base_inode, shuffle_index=i)
