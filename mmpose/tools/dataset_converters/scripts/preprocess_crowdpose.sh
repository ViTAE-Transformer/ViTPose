#!/usr/bin/env bash

DOWNLOAD_DIR=$1
DATA_ROOT=$2

tar -zxvf $DOWNLOAD_DIR/OpenDataLab___CrowdPose/raw/CrowdPose.tar.gz -C $DATA_ROOT
rm -rf $DOWNLOAD_DIR/OpenDataLab___CrowdPose
