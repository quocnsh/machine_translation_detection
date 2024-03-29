# Machine Translation Text Detection

# Requirement
- Tensorflow 1.14
- Torch 1.4.0
- Fairseq 0.9.0

# Pretrain data
- Download [BERT-Base pretrained model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) and put in the folder: `bert/uncased_L-12_H-768_A-12`

# Running 
- Training with one language: 
`python run_generator.py --train_languages de --test_language de`
- Training with multiple languages: 
`python run_generator.py --train_languages ru,de --test_language ru`
# Support
- The supported languages are `ru`(Russian) and `de`(German)

# Acknowledgments
- Code refer to: [BERT](https://github.com/google-research/bert).
