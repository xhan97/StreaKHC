# StreaKHC #
**StreaKHC**  is a novel incremental hierarchical clus-tering algorithm for efficiently mining massive streaming data. It utilises a top-down search strategy to identify the deepest node that has the point-set similarity to the new point greater than or equal to a given similarity threshold, and then insert the new point into this node and its all parents. Technical details and analysis of the algorithm can be found in paper.


## Setup ##

If running the python code, download and Install Anaconda's Python3

```
https://docs.continuum.io/anaconda/install
```

If running python code, install numba

```
conda install numba
```

Set environment variables:

```
source bin/setup.sh
```

If want to visulize the build tree, install Graphviz

```
sudo apt install graphviz
```

## Run test ##

Run test on data set:
```
 ./bin/run_evalation.sh
```
The experiment result is shown in /exp_out/ defautly.


## Notes ##

  - If you do not need to visualize the generated tree, you can comment out the corresponding code in the /bin/run_evaluation.sh.
  - Perl is used to shuffle the data.You'll need perl installed on your system to run experiment shell scripts.  If you can't run perl, you can change this to another shuffling method of your choice.
  - The scripts in this project use environment variables set in the setup script. You'll need to source this set up script in each shell session running this project.
