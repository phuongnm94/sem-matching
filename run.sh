# install env
conda create -n env_sem_matching python=3.8
conda activate env_sem_matching
pip install -r sp_libs/IDSL-semparser/baseline/requirements.txt

# install env

cd sp_libs/IDSL-semparser/baseline
mkdir outputs/
mkdir outputs/top/
mkdir  outputs/top/trainer_baseline 

# train model
bash semparser_baseline.sh

# run server 
python ./server.py ./outputs/top/trainer_baseline/epoch=5.ckpt ./outputs/top/trainer_baseline/lightning_logs/version_0/hparams.yaml

# access to server at: http://localhost:7009/