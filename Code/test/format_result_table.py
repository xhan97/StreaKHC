'''
Author: your name
Date: 2021-01-17 12:06:42
LastEditTime: 2021-05-14 00:41:30
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \StreamHC\Code\test\format_result_table.py
'''
import numpy as np
import sys

def load_result_file(fn):
    """
    Result file format:
        algorithm <tab> dataset <tab> dendrogram purity

    Args:
        fn: filename

    Returns: dictionary: alg -> dataset -> (mean(dp),std(dp)

    """
    alg2dataset2score = {}
    with open(fn,'r') as fin:
        for line in fin:
            try:
                splt = line.strip().split("\t")
                alg,dataset,dp = splt
                if dataset not in alg2dataset2score:
                    alg2dataset2score[dataset] = {}
                if alg not in alg2dataset2score[dataset]:
                    alg2dataset2score[dataset][alg] = []
                alg2dataset2score[dataset][alg].append(float(dp))
            except:
                pass

    for dataset in alg2dataset2score:
        for alg in alg2dataset2score[dataset]:
            mean = np.mean(alg2dataset2score[dataset][alg])
            std = np.std(alg2dataset2score[dataset][alg])
            alg2dataset2score[dataset][alg] = (mean,std)

    return alg2dataset2score

def escape_latex(s):
    s = s.replace("_","\\_")
    return s

def latex_table(alg2dataset2score):
    table_string = """\\begin{table}\n\\begin{center}\n\\begin{tabular}"""
    num_ds = max([len(alg2dataset2score[x]) for x in alg2dataset2score])
    formatting = "{c" + "c" * num_ds + "}"
    table_string += format(formatting)
    table_string += "\n\\toprule\n"
    ds_names = list(set([name for x in alg2dataset2score for name in alg2dataset2score[x]]))
    table_string += "\\bf Dataset & \\bf " + " & \\bf ".join([escape_latex(x) for x in ds_names]) + "\\\\\n"
    table_string += "\\midrule\n"
    alg_names = alg2dataset2score.keys()
    alg_names = sorted(alg_names)
    for alg in alg_names:
        scores = [ "%.2f $\\pm$ %.2f" % (alg2dataset2score[alg][ds][0],alg2dataset2score[alg][ds][1]) if ds in alg2dataset2score[alg] else "-" for ds in ds_names]
        table_string += "%s & %s \\\\\n" % (alg," & ".join(scores))
    table_string += "\\bottomrule\n\\end{tabular}\n\\end{center}\n\\end{table}"
    return table_string


if __name__ == "__main__":
    print(latex_table(load_result_file("newalldata.csv")))