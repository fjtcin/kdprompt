# kdprompt

```bash
python autoencoder.py --linear
python train_teacher.py --dataset ogbn-arxiv --prompts_dim 64
python train_student.py --dataset ogbn-arxiv --prompts_dim 64 --max_epoch 500 --patience 50 --save_results --lamb .5  # default lamb=0 (aka all GNN)
python test_prompt.py --dataset cora --prompts_dim 64 --max_epoch 500000 --patience 5000
```

```bash
# Baseline
python train_teacher.py --exp_setting tran --teacher SAGE --dataset cora
python train_teacher.py --exp_setting tran --teacher MLP --dataset cora
```
