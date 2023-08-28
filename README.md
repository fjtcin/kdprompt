# kdprompt

```bash
python autoencoder.py --dropout_autoencoder 0.7 --dropout_MLP 0.3
python train_teacher.py --dataset ogbn-arxiv --prompts_dim 64 --seed
python train_student.py --dataset ogbn-arxiv --student MLP --prompts_dim 64 --patience 20 --save_results --lamb .5 --seed   # default lamb=0 (aka all GNN)
python test_prompt.py --dataset cora --student MLP --prompts_dim 64 --seed
```

```bash
# Baseline
python train_teacher.py --exp_setting tran --teacher SAGE --dataset cora
python train_teacher.py --exp_setting tran --teacher MLP --dataset cora
```
