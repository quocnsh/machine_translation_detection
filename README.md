# Machine Translation Text Detection

# Requirement:
- tensorflow 1.14
- torch 1.4.0
- fairseq 0.9.0

# Pretrain data
- Download BERT-Base pretrained model and put in ./bert/uncased_L-12_H-768_A-12 folder

# Running 
- run file with following parameter
- python run_generator.py --train_languages de,ru --test_language de

# Support
- The supported languages are de (German) and ru (Russian)

