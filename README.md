# kdprompt

```bash
python autoencoder.py --linear
python train_teacher.py --dataset ogbn-arxiv
```

```bash
python train_teacher.py --exp_setting tran --teacher SAGE --dataset ogbn-arxiv
python train_student.py --exp_setting tran --teacher SAGE --student MLP --dataset ogbn-arxiv --out_t_path outputs

# GNN
cat /home/fjtcin/Documents/git/kdprompt/outputs/transductive/ogbn-arxiv/SAGE/exp_results
cat /home/fjtcin/Documents/git/kdprompt/outputs/transductive/ogbn-arxiv/SAGE/seed_0/log

# MLP
cat /home/fjtcin/Documents/git/kdprompt/outputs/transductive/ogbn-arxiv/SAGE_MLP/exp_results
cat /home/fjtcin/Documents/git/kdprompt/outputs/transductive/ogbn-arxiv/SAGE_MLP/seed_0/log
```

```bash
python train_teacher.py
python train_student.py --max_epoch 100000 --patience 1000 --save_results --lamb .5
python test_prompt.py --max_epoch 100000 --patience 1000
```
