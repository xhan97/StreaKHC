#!/usr/bin/env bash

set -exu

$SKHUSTER_ROOT/bin/data_processing/download_aloi.sh
$SKHUSTER_ROOT/bin/data_processing/download_glass.sh