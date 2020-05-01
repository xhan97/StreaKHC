#!/usr/bin/env bash

set -xu

mkdir $XCLUSTER_ROOT/dep
pushd $XCLUSTER_ROOT/dep
wget http://www.gtlib.gatech.edu/pub/apache/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz
tar -xvf apache-maven-3.6.3-bin.tar.gz
popd
