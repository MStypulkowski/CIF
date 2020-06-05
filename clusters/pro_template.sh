#!/bin/bash

# Script for the Prometheus slurm cluster

set -x

RVERB=""  # =-v

REMOTE_USER=plgjch
REMOTE_HOST=pro.cyfronet.pl

# location of the main repository (contains data/)
REMOTE_CIF_DATA_DIR=/net/people/plgjch/scratch/CIF_DATA
REMOTE_MINICONDA_DIR=/net/archive/groups/plggneurony/os/miniconda3

# top-level directory for experiments
REMOTE_EXPERIMENT_RUNDIR=/net/people/${REMOTE_USER}/scratch/cif_experiments

# adjust the main loop
for BSIZE in 1 2; do

# Step 1: set param names
# low-level directory for experiments
EXP_TAG=SET_BS
NAME=BS${BSIZE}
DIR=$EXP_TAG/$NAME
EXP_DIR=$REMOTE_EXPERIMENT_RUNDIR/$DIR

TMP_DIR=`mktemp -d`
mkdir $TMP_DIR/code

echo "Setting up experiment in $TMP_DIR"

cat > $TMP_DIR/exp_config.yaml <<EOF
root_dir: '$REMOTE_CIF_DATA_DIR/ShapeNetCore.v2.PC15k'
save_models_dir: '$EXP_DIR/saves/'
save_models_dir_backup: '$EXP_DIR/backup/'
load_models_dir: '$EXP_DIR/saves/'
# data_dir: ''
losses: '$EXP_DIR/losses.txt'

init_w: False # set True unless precomputed
use_EMD: False # False - Chamfer Distance
load_models: False # set False unless pretrained
use_new_f: True
use_new_g: True
use_random_dataloader: False

batch_size: $BSIZE
batch_size_if_random_split: 3000
n_epochs: 1000
n_neurons: 512
n_flows_F: 5
n_flows_G: 3
l_rate: 0.0001
emb_dim: 64
x_noise: 0.0001
w_noise: 0.02
prior_z_var: 1.0
prior_e_var: 1.0

# for test reconstruction - embeddings training
# embs_dir: ''
load_embs: False
l_rate4recon: 0.1
id4recon: 3

# post training tuning
train_F: True
train_G: True

####### PointFlow-related args #######

# these two does not influence the result, they're neede for comptability
# with point flow loaders
tr_sample_size: 1024
te_sample_size: 1024
#

normalize_per_shape: false
normalize_std_per_axis: false
scale: 1.0
categories: ["chair"]
EOF

cat > $TMP_DIR/exp_train.sh <<EOF
#!/bin/bash -l
## Job name
#SBATCH -J ${EXP_TAG}_${NAME}
## Nodes
#SBATCH -N 1
## CPU per Node
#SBATCH -c 4
## GPU
#SBATCH --gres=gpu:2
##
#SBATCH --mem=36GB
##
#SBATCH --time=72:00:00
##
#SBATCH -A asrcontinuatoin
##
#SBATCH -p plgrid-gpu
##
#SBATCH --output="$EXP_DIR/exp_%j.out"
##
#SBATCH --error="$EXP_DIR/exp_%j.out"
## go to the exp dir
cd "$EXP_DIR"
/bin/hostname
eval "\$($REMOTE_MINICONDA_DIR/bin/conda shell.bash hook)"
conda activate cif
export PYTHONPATH=$EXP_DIR/code
python code/experiments/train/train_model.py --config exp_config.yaml
EOF

# copy our current dir's contents
rsync --exclude '.*' \
      --exclude '*.pyc' \
      --exclude '*.ipynb' \
      --filter=':- .gitignore' \
    $RVERB -lrpt `pwd`/ $TMP_DIR/code

# link in the MDS weights
mkdir $TMP_DIR/saves
ln -s ${REMOTE_CIF_DATA_DIR}/w.pth $TMP_DIR/saves/

ssh -q $REMOTE_USER@$REMOTE_HOST mkdir -p $EXP_DIR

# Transmit the code and scripts
rsync $RVERB -lrpt -e "ssh -q" $TMP_DIR/ $REMOTE_USER@$REMOTE_HOST:$EXP_DIR/

ssh -q $REMOTE_USER@$REMOTE_HOST sbatch \
    `#--gres="" --time=00:10:00 -p plgrid-testing` \
    $EXP_DIR/exp_train.sh

rm -Rf $TMP_DIR

done

echo "Queue status"
ssh -q $REMOTE_USER@$REMOTE_HOST squeue