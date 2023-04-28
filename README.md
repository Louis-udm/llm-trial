# Large Language Models Trials

# .env
PYTHONPATH=.:${PYTHONPATH}
TRANSFORMERS_OFFLINE=1

# cmds
python -u trials/vgcn_bert/train_cola.py > /log/vb_train_cola_20230428_transparent.log 2>&1 &
tail -f /log/vb_train_cola_20230428_transparent.log