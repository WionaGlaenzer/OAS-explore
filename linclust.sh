mkdir linclust
cd linclust
echo "path: $(pwd)"
mmseqs createdb ../outputs/sequences.fasta antibody_DB
mmseqs linclust --cov-mode 1 -c 0.9 --min-seq-id 0.9 antibody_DB antibody_DB_clu tmp
mmseqs createsubdb antibody_DB_clu antibody_DB antibody_DB_clu_rep
mmseqs convert2fasta antibody_DB_clu_rep antibody_DB_clu_rep.fasta

