#!/usr/bin/env bash

set -exu

pushd $SKHUSTER_ROOT
mvn clean package
popd