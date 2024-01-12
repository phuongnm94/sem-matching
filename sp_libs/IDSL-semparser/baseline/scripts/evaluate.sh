REPO_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

MODEL_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams/outputs/top/trainer_sisemroberta_5_epochs/epoch=10.ckpt
HPARAM_PATH=/home/s2110418/Master/Semantic_parsing/IDSL-semparser/semparser_finetuning_hyperparams/outputs/top/trainer_sisemroberta_5_epochs/lightning_logs/version_0/hparams.yaml

python evaluate/evaluate.py $MODEL_PATH $HPARAM_PATH