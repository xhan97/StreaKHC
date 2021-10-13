#!/usr/bin/env bash

export STREASKH_ROOT=`pwd`
export STREASKH_data=$STREASKH_ROOT/data
export XCLUSTER_JARPATH=$SKHUSTER_ROOT/target/xcluster-0.1-SNAPSHOT-jar-with-dependencies.jar
export PYTHONPATH=$STREASKH_ROOT/src/python:$PYTHONPATH
export PATH=$STREASKH_ROOT/dep/apache-maven-3.6.1/bin:$PATH

if [ ! -f $STREASKH_ROOT/.gitignore ]; then
    echo ".gitignore" > $STREASKH_ROOT/.gitignore
    echo "target" >> $STREASKH_ROOT/.gitignore
    echo ".idea" >> $STREASKH_ROOT/.gitignore
    echo "__pycache__" >> $STREASKH_ROOT/.gitignore
    echo "dep" >> $STREASKH_ROOT/.gitignore
    echo "data" >> $STREASKH_ROOT/.gitignore
    echo "test_out" >> $STREASKH_ROOT/.gitignore
    echo "experiments_out" >> $STREASKH_ROOT/.gitignore
    echo ".DS_STORE" >> $STREASKH_ROOT/.gitignore
    echo "*.iml" >> $STREASKH_ROOT/.gitignore
fi
