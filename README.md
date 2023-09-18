# kdprompt

```bash
python autoencoder.py --dropout_autoencoder 0.7 --dropout_MLP 0.3
python train_teacher.py --dataset ogbn-arxiv --prompts_dim 64 --seed
python train_student.py --dataset ogbn-arxiv --student MLP --prompts_dim 64 --patience 20 --save_results --feature_aug_k 1 --upstream_feature_aug_k 0 --lamb .5 --seed   # default lamb=0 (aka all GNN)
python test_prompt.py --dataset cora --student MLP --prompts_dim 64 --labelrate_train 20 --feature_aug_k 1 --upstream_feature_aug_k 1 --seed
```

```bash
# SAGE
python train_teacher.py --dataset ogbn-arxiv --teacher SAGE --prompts_dim 64 --save_results --seed 10
python test_prompt.py --dataset cora --teacher SAGE --prompts_dim 64 --seed 10
# MLP
python train_teacher.py --dataset ogbn-arxiv --teacher MLP --prompts_dim 64 --save_results --seed 10
python test_prompt.py --dataset cora --teacher MLP --prompts_dim 64 --seed 10
```

```bash
# Baseline
python train_teacher.py --exp_setting tran --teacher SAGE --dataset cora --labelrate_train 20 --feature_aug_k 0
python train_teacher.py --exp_setting tran --teacher MLP --dataset cora --labelrate_train 20 --feature_aug_k 1
```
