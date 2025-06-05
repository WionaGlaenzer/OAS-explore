#!/bin/bash
linclust_dir="$1"
sequences_fasta="$2"
similarity="$3"
coverage="$4"
mkdir $linclust_dir
cd $linclust_dir
echo "path: $(pwd)"
mmseqs createdb $sequences_fasta antibody_DB
mmseqs linclust --cov-mode 1 -c $coverage --min-seq-id $similarity antibody_DB antibody_DB_clu tmp
mmseqs createsubdb antibody_DB_clu antibody_DB antibody_DB_clu_rep
mmseqs convert2fasta antibody_DB_clu_rep antibody_DB_clu_rep.fasta