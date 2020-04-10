# Machine Translation Text Detection

# Requirement:
- tensorflow 1.14
- torch 1.4.0
- fairseq 0.9.0

# Pretrain data
- Download BERT-Base pretrained model and put in the folder: bert/uncased_L-12_H-768_A-12

# Running 
- python run_generator.py --train_languages de,ru --test_language de

# Support
- The supported languages are de (German) and ru (Russian)
