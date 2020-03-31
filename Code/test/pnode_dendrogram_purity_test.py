"""
Copyright (C) 2017 University of Massachusetts Amherst.
This file is part of "xcluster"
http://github.com/iesl/xcluster
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from scipy.spatial.distance import cdist
from models.INode import INode
from utils.dendrogram_purity import dendrogram_purity
from utils.deltasep_utils import create_dataset
from utils.add_nne import addNNE,aNNE_similarity
from utils.Graphviz import Graphviz


def add_nne_data(dataset,n,psi,t):
  """Add ik value to dataset.
  Args:
    dataset - a list of points with which to build the tree.
    n - the number of dataset to build aNNE metrix
    psi - parameter of ik
    t - paremeter of ik
  Return:
    dataset with ik value
    
  """
  
  
  met = [pt[0] for pt in dataset[:n]]
  
  x = cdist(met,met, 'euclidean') 
  oneHot,subIndexSet,aNNEMetrix = aNNE_similarity(x,psi,t)
  for i, pt in enumerate(dataset[:n]):
      pt.append(aNNEMetrix[i])
      
  return oneHot,subIndexSet,dataset
  

def create_trees_w_purity_check(n,psi,t,dataset):
    """Create trees over the same points.

    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.  After each insertion, verify that the dendrogram purity
    is still 1.0 (perfect).

    Args:
        dataset - a list of points with which to build the tree.

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """
    np.random.shuffle(dataset)
    met = [pt[0] for pt in dataset[:n]]
    
    oneHot,subIndexSet,dataset = add_nne_data(dataset,n,psi,t)
    root = INode(exact_dist_thres=10)
    
    for i, pt in enumerate(dataset[:n]):
        if len(pt)==3:
          ikv = addNNE(met,pt[0],oneHot,subIndexSet)
          pt.append(ikv)
        root = root.insert(pt, collapsibles=None, L= float('inf'))
        
#        if i % 30 == 0:
#            assert(dendrogram_purity(root) == 1.0)
#            assert(root.point_counter == (i + 1))
    return root


if __name__ == '__main__':
    """Test that PERCH produces perfect trees when data is separable.

    PERCH is guarnateed to produce trees with perfect dendrogram purity if
    thet data being clustered is separable. Here, we generate random separable
    data and run PERCH clustering.  Every 10 points assert that the purity is
    1.0."""
    dimensions = [5]
    size = 30
    num_clus = 5
    n=100
    psi=10
    t = 200
    
    for dim in dimensions:
        print("TESTING DIMENSIONS == %d" % dim)
        dataset = create_dataset(dim, size, num_clusters=num_clus)                                         
        
    root = create_trees_w_purity_check(n,psi,t,dataset)
    print(dendrogram_purity(root))
    
    graph = Graphviz()
    graph.write_tree('tree.txt',root)
