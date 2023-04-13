#!/bin/sh
mkdir -p dataset/rt_bene/archive
mkdir -p dataset/rt_bene/raw

# Download & Setup RT-BENE
for i in $(seq -f "%03g" 0 16)
do
   wget -O "dataset/rt_bene/archive/s${i}_ng.tar"  "https://zenodo.org/record/2529036/files/s${i}_noglasses.tar?download=1"
   tar -xf "dataset/rt_bene/archive/s${i}_ng.tar" -C dataset/rt_bene/raw
   wget -O "dataset/rt_bene/raw/s${i}_noglasses/blink_labels.csv" "https://zenodo.org/record/3685316/files/s${i}_blink_labels.csv?download=1"
done

# Download & Setup RT-GENE
# for i in $(seq -f "%03g" 0 16)
# do
#    wget -O "dataset/rt_bene/archive/s${i}.tar"  "https://zenodo.org/record/2529036/files/s${i}_glasses.tar?download=1"
#    tar -xf "dataset/rt_bene/archive/s${i}.tar" -C dataset/rt_bene/raw
# done
