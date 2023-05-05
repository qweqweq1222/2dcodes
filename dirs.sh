#!/bin/sh
mkdir qr
mkdir dm 
mkdir -p results/plots
mkdir -p results/images
mkdir -p results/images/wrong  
mkdir -p results/images/all  
mkdir -p test_dataset/images
mkdir -p test_dataset/labels
mkdir -p results/images/wrong/qr
mkdir -p results/images/wrong/dm
mkdir -p results/stats_txt
unzip qr_codes.zip -d qr/
unzip dm_codes.zip -d dm/
cp qr/train/images/* test_dataset/images
cp qr/train/labels/* test_dataset/labels
cp dm/train/images/* test_dataset/images
cp dm/train/labels/* test_dataset/labels