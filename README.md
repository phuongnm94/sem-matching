# sem-matching

## env setup


1. Install packages
    ```
    conda create -n env_sem_matching python=3.8
    conda activate env_sem_matching
    pip install -r sp_libs/IDSL-semparser/baseline/requirements.txt
    ```
## run 

1. activate env 
   ```
   conda activate env_sem_matching
   ```
2. train a sem parser model 
    ```cmd
    cd sp_libs/IDSL-semparser/baseline/scripts 
    bash semparser_baseline.sh
    ```
3. run server 
    ```cmd 
    cd sp_libs/IDSL-semparser/baseline/ && \
    python ./server.py ./outputs/top/trainer_baseline/epoch=5.ckpt ./outputs/top/trainer_baseline/lightning_logs/version_0/hparams.yaml
    ```
