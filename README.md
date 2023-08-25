# kdprompt

```bash
python autoencoder.py --dropout_autoencoder 0.7 --dropout_MLP 0.3

python train_teacher.py --teacher MLP --dataset ogbn-arxiv --prompts_dim 64 --save_results
python test_prompt.py --teacher MLP --dataset cora --prompts_dim 64 --max_epoch 50000 --patience 5000

python train_teacher.py --teacher SAGE --dataset ogbn-arxiv --prompts_dim 64 --save_results
python test_prompt.py --teacher SAGE --dataset cora --prompts_dim 64 --max_epoch 5000 --patience 500
```

```bash
# Baseline
python train_teacher.py --exp_setting tran --teacher SAGE --dataset cora
python train_teacher.py --exp_setting tran --teacher MLP --dataset cora
```
