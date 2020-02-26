# machine_translation_detection

# Requirement:
- tensorflow 1.14
- torch 1.4.0
- fairseq 0.9.0

# pretrain data
- Download BERT-Base pretrained model and put in ./bert/uncased_L-12_H-768_A-12 folder

# run 
- run file with following parameter
python run_generator.py --train_languages de,ru --test_language de

The language support includes de (German) and ru (Russian)

