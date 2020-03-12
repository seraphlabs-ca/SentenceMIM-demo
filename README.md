# SentenceMIM-demo

This repo contains code to reproduce some of the results presented in the paper ["SentenceMIM: A Latent Variable Language Model"](https://arxiv.org/abs/2003.02645)

Code is based on: ```https://github.com/timbmg/Sentence-VAE```.

# Experiments

## Training 

```
hidden_size=512
embedding_size=300
dataset=ptb
for latent_size in 16 128 512; do
    # MIM
    ./train.py  \
        --save_model_path exp \
        --test \
        --dataset ${dataset}  \
        --embedding_dropout 0.5 \
        --epochs 200 \
        --max_sequence_length 100 \
        --batch_size 20 \
        --embedding_size ${embedding_size} \
        --hidden_size ${hidden_size} \
        --latent_size ${latent_size} \
        --optim adam \
        -lr 0.001 \
        -x0 0 \
        --mim
    
    # VAE
    for x0 in 0 10000; do
        ./train.py  \
            --save_model_path exp \
            --test \
            --dataset ${dataset}  \
            --embedding_dropout 0.5 \
            --epochs 200 \
            --max_sequence_length 100 \
            --batch_size 20 \
            --embedding_size ${embedding_size} \
            --hidden_size ${hidden_size} \
            --latent_size ${latent_size} \
            --optim adam \
            -lr 0.001 \
            -x0 ${x0}
    done

    # AE
    ./train.py  \
        --save_model_path exp \
        --test \
        --dataset ${dataset}  \
        --embedding_dropout 0.5 \
        --epochs 200 \
        --max_sequence_length 100 \
        --batch_size 20 \
        --embedding_size ${embedding_size} \
        --hidden_size ${hidden_size} \
        --latent_size ${latent_size} \
        --optim adam \
        -lr 0.001 \
        --marginal \
        -x0 0
done
```

## Testing 

```
./test.py --max_sequence_length 100 --test_epochs 1 --seed 1 -maxsl 20 --batch_size 20 --temperature 1.0 --test --split test  bin/ptb_2019-Nov-26_18-45-49_mim
```

# Citation

Please cite using the following bibtex entry

```
@ARTICLE{2020arXiv200302645L,
       author = {{Livne}, Micha and {Swersky}, Kevin and {Fleet}, David J.},
        title = "{SentenceMIM: A Latent Variable Language Model}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language, Computer Science - Machine Learning, Statistics - Machine Learning, 68T50, I.2.7},
         year = 2020,
        month = feb,
          eid = {arXiv:2003.02645},
archivePrefix = {arXiv},
       eprint = {2003.02645},
 primaryClass = {cs.CL},
}
```
