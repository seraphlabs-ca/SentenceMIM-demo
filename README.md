# SentenceMIM-demo

This repo contains code to reproduce some of the results presented in the paper ["SentenceMIM: A Latent Variable Language Model"](https://arxiv.org/abs/2003.02645).

Code is based on: <https://github.com/timbmg/Sentence-VAE>.

## Installation

All data is included in repo.
You can install the required packages with the command below.
NOTE: Virtual env is highly recommended to prevent corruption of your personal pip environment.

```
# Use at your own risk - virtual env is highly recommended
pip install requirements.txt 
```

# Experiments

Below is summary of the results, and the corresponding commands to train and test MIM, VAE, and AE (with similar architecture) over Penn Treebank (PTB) dataset.

## Dataset

|              | Sentences |        |       | Tokens |       Words per Sentence     |
|--------------|:---------:|:------:|:-----:|:------:|:----------------:|
|              |   Train   | Valid. |  Test | Vocab. |     #w (max)     |
| PTB          |    3370   |  3761  | 42068 |  10000 | 21 +/- 10 (82) |


## Results Summary


### Perplexity

NLL/PPL results are an upper bound, computed with MELBO for MIM, and ELBO for VAE.

| Model      | PPL    | NLL    | BLEU   | Parameters |
|------------|--------|--------|--------|-----------:|
| AE (16)    |        |        | 0.2916 |        11M |
| AE (128)   |        |        | 0.5321 |        11M |
| AE (512)   |        |        | 0.5745 |        12M |
| sVAE (16)  | 113.14 | 107.38 | 0.0882 |        11M |
| sVAE (128) | 117.2  | 108.19 | 0.0803 |        11M |
| sVAE (512) | 121.7  | 109.04 | 0.0809 |        12M |
| sMIM (16)  | 75.29  | 98.14  | 0.2975 |        11M |
| sMIM (128) | 28.58  | 76.14  | 0.5542 |        11M |
| sMIM (512) | **19.02**  | **66.9**   | **0.6226** |        12M |


### Reconstruction 

Below are reconstruction results for best performing models.

**sVAE (16)**
```
DATA: <sos> the system is the problem not an individual member
MEAN RECON: they worry about their or ceiling at their own <unk> wellcome sears ' s not forget down <eos>
Z RECON: often con artists can ignore their credit judgments just how we had with them who now increase after <unk> losing <unk> <unk> and now discarded our country <eos>
Z PERT: stock-index arbitrage volatility volatility volatility volatility volatility portfolio portfolio portfolio portfolio portfolio volatility volatility portfolio portfolio portfolio portfolio portfolio portfolio portfolio portfolio portfolio portfolio portfolio portfolio portfolio ...

DATA: <sos> sony itself declines to comment
MEAN RECON: hyundai and other cities have been asked to exclude <eos>
Z RECON: brian <eos>
Z PERT: mcdonnell douglas turner turner turner pictures mother singer a genetic <unk> gene ogilvy former sanford <unk> turner pictures woman writer turner broadcasting university gene turner turner pictures <unk> <unk> university in genetic damage but her ted district was buddy and jazz <unk> in cells never complex kate judge o'kicki and certified mother <unk> turner pictures <unk> <unk> st . louis park <eos>
```

**sMIM (512)**
```
DATA: <sos> the system is the problem not an individual member
MEAN RECON: the system is not the problem an individual member <eos>
Z RECON: the system is the problem not an individual member <eos>
Z PERT: the system is not the problem an individual member <eos>


DATA: <sos> sony itself declines to comment
MEAN RECON: sony itself declines to comment <eos>
Z RECON: sony industry declines to comment <eos>
Z PERT: sony itself sony itself to n <eos>
```

### Interpolation 

Below are interpolation results for best performing models.
Sentences in ```[ brackets ]``` are the source, and destination sentences.

**sVAE (16)**
```
[ <sos> the system is the problem not an individual member ]

all the programs and the people party held me how are set up leads and by boosting their earthquake he says <eos>

apparently she says the government ' s perfectly is born nervous that other <unk> chinese <unk> and lewis <unk> <unk> <unk> and psychiatric publishers such as brooks brothers brothers into leather ' s landmark the affiliates are an behavior for them <eos>

the combination they are bearing a new <unk> approach company can find anything that customers are n't registered of them by using the so-called global office to understand a balance of <unk> out of the information ' s health-care system <eos>

as much as n n it does n't think they should ask now from facing paying their work <eos>

one participant probably change in congressional policy was then at the same time the <unk> yesterday was sending by the <unk> ' s cat of management and his own colleagues who joined the work force of <unk> <eos>

<unk> is perfect in the state <eos>

this is as part of heart on the acquisition of the no . n marketplace hoped that everyone does n't believe that the british attorney reported a ruling at its forecast mr . <unk> has earlier a meeting with a company something that money he said in october <eos>

there is included that its number of bonds measures activity are still much less priced to yield to treasurys <eos>

tucson creating alcohol <eos>

the most form spacecraft reading in advertising of prime minister mr . watson <unk> the most dramatic attempt into a vast region <eos>

[ <sos> sony itself declines to comment ]
```

**sMIM (512)**
```
[ <sos> the system is the problem not an individual member ]

the problem is not changing the an insider problem <eos>

the system is not the problem an individual problem <eos>

the problem is not the problem on an individual merits <eos>

the system system is not an individual member <eos>

the system itself is not an immediate <eos>

sony itself the problem is n't difficult to work <eos>

columbia corp . never put the voice to <eos>

sony corp . almost to comment the problem <eos>

sony itself declines almost to comment <eos>

sony itself declines to comment <eos>

[ <sos> sony itself declines to comment ]
```

### Sampling 

Below are sampling results for best performing models.

**sVAE (16)**
```
SAMP: the report suggested no longer fraud was <unk> but they ' re liable to see a full compromise of confidence in north america <eos>

SAMP: the investor set aside tuesday ' s <unk> program trading as a plant who had moved to make n million in the past several months <eos>

SAMP: first meridian the government can be <unk> out of a time in australia who is now to emerge his own affidavit <eos>
```

**sMIM (512)**
```
SAMP: the report suggested no longer fraud was <unk> but they ' re liable to see a full compromise of confidence in north america <eos>

SAMP: the investor set aside tuesday ' s <unk> program trading as a plant who had moved to make n million in the past several months <eos>

SAMP: first meridian the government can be <unk> out of a time in australia who is now to emerge his own affidavit <eos>
```


## Training Commands

```
hidden_size=512
embedding_size=300
dataset=ptb
for latent_size in 16 128 512; do
    # MIM
    ./train.py  \
        --seed 1 \
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
            --seed 1 \
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
        --seed 1 \
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

## Testing Commands

NLL is upper bounded using MELBO for MIM, and ELBO for VAE.

```
./test.py \
    --max_sequence_length 100 \
    --test_epochs 1 \
    --seed 1 \
    --max_sample_length 20 \
    --batch_size 20 \
    --test \
    --split test \
    --test_bleu \
    --test_sample \
    --test_interp \
    data/torch-generated/exp/*
```

All results will be store in the corresponding results path in ```best-test-1-marginal0-mcmc0.txt``` (name corresponds the command line arguments used with ```test.py``` script).

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
