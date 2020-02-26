import os
import random
import csv
import nltk
nltk.download('punkt')
import torch
import sys
import shutil

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model')


en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model')

data_file = r'./input/data.txt'
dev_rate = 0.1
test_rate = 0.1
SEED = 0

def read_all_lines(input_file):
    fi = open(input_file,"r", encoding="utf8") 
    lines = fi.readlines()
    fi.close()
    return lines

def translate(item, first_language, second_language):
    if (first_language == 'en' and second_language == 'de'):
        return en2de.translate(item, beam=5)
    if (first_language == 'de' and second_language == 'en'):
        return de2en.translate(item, beam=5)
    if (first_language == 'en' and second_language == 'ru'):
        return en2ru.translate(item, beam=5)
    if (first_language == 'ru' and second_language == 'en'):
        return ru2en.translate(item, beam=5)
#    return "%s_%s_%s"%(item, first_language, second_language)

#def translate(item, first_language, second_language):
#    return "%s_%s_%s"%(item, first_language, second_language)
    

def generate_round_trip(item, language):
    first_translation = translate(item, 'en', language)
    second_translation = translate(first_translation, language, 'en')
    return second_translation
    
def generate_round_trips(data, language):
    result = []
    for item in data:
        result.append(generate_round_trip(item, language))
    return result                
    
def write_all_lines(data, output_file):
    fo = open(output_file,"w+", encoding="utf8") 
    for item in data:
        fo.write(item+"\n")
    fo.close()
    
def generate_train_dev(process_folder, data, train_language, belong):
    machine_data = generate_round_trips(data, train_language)
    round_trip_machine_data = generate_round_trips(machine_data, train_language)
    output_file = os.path.join(process_folder, "%s_human.txt"%belong)
    write_all_lines(data, output_file)
    output_file = os.path.join(process_folder, "%s_human_BT.txt"%belong)
    write_all_lines(machine_data, output_file)
    output_file = os.path.join(process_folder, "%s_machine.txt"%belong)
    write_all_lines(machine_data, output_file)
    output_file = os.path.join(process_folder, "%s_machine_BT.txt"%belong)
    write_all_lines(round_trip_machine_data, output_file)
    
    
def generate_test(process_folder, test_data, train_language, test_language):
    machine_data = generate_round_trips(test_data, test_language)
    round_trip_human_data = generate_round_trips(test_data, train_language)
    round_trip_machine_data = generate_round_trips(machine_data, train_language)
    output_file = os.path.join(process_folder, "test_human.txt")
    write_all_lines(test_data, output_file)
    output_file = os.path.join(process_folder, "test_human_BT.txt")
    write_all_lines(round_trip_human_data, output_file)
    output_file = os.path.join(process_folder, "test_machine.txt")
    write_all_lines(machine_data, output_file)
    output_file = os.path.join(process_folder, "test_machine_BT.txt")
    write_all_lines(round_trip_machine_data, output_file)   
    
def generation_train_dev_format(human, human_BT, machine, machine_BT, output, shuffle=False):    
    results = []
    results.append('Quality\t#1 ID\t#2 ID\t#1 String\t#2 String\n')                                            
    human_texts = read_all_lines(human)
    human_BT_texts = read_all_lines(human_BT)
    machine_texts = read_all_lines(machine)
    machine_BT_texts = read_all_lines(machine_BT)
    
    if shuffle == True:
        for idx, text in enumerate(human_texts):
            results.append("0\t\t\t%s\t%s\n"%(human_texts[idx].strip(),human_BT_texts[idx].strip()))
            results.append("1\t\t\t%s\t%s\n"%(machine_texts[idx].strip(),machine_BT_texts[idx].strip()))
    else:            
        for idx, text in enumerate(human_texts):
            results.append("0\t\t\t%s\t%s\n"%(human_texts[idx].strip(),human_BT_texts[idx].strip()))
                       
        for idx, text in enumerate(machine_texts):
            results.append("1\t\t\t%s\t%s\n"%(machine_texts[idx].strip(),machine_BT_texts[idx].strip()))                   
    fo = open(output,"w+", encoding="utf8") 
    fo.writelines(results)
    fo.close()


    
def generation_test_format(human, human_BT, machine, machine_BT, output, shuffle = False):
    results = []
    results.append('index\t#1 ID\t#2 ID\t#1 String\t#2 String\n')                                            
    human_texts = read_all_lines(human)
    human_BT_texts = read_all_lines(human_BT)
    machine_texts = read_all_lines(machine)
    machine_BT_texts = read_all_lines(machine_BT)
    
    if shuffle == True:
        for idx, text in enumerate(human_texts):
            results.append("\t\t\t%s\t%s\n"%(human_texts[idx].strip(),human_BT_texts[idx].strip()))
            results.append("\t\t\t%s\t%s\n"%(machine_texts[idx].strip(),machine_BT_texts[idx].strip()))
    else:
        for idx, text in enumerate(human_texts):
            results.append("\t\t\t%s\t%s\n"%(human_texts[idx].strip(),human_BT_texts[idx].strip()))
        for idx, text in enumerate(machine_texts):
            results.append("\t\t\t%s\t%s\n"%(machine_texts[idx].strip(),machine_BT_texts[idx].strip()))
                   
    fo = open(output,"w+", encoding="utf8") 
    fo.writelines(results)
    fo.close()


def generate_MRPC(input_folder):
    test_human = os.path.join(input_folder,"test_human.txt")
    test_human_BT = os.path.join(input_folder, "test_human_BT.txt")
    test_machine = os.path.join(input_folder,"test_machine.txt")
    test_machine_BT = os.path.join(input_folder,"test_machine_BT.txt")

    dev_human = os.path.join(input_folder,"dev_human.txt")
    dev_human_BT = os.path.join(input_folder,"dev_human_BT.txt")
    dev_machine = os.path.join(input_folder, "dev_machine.txt")
    dev_machine_BT = os.path.join(input_folder,"dev_machine_BT.txt")

    
    train_human = os.path.join(input_folder,"train_human.txt")
    train_human_BT = os.path.join(input_folder,"train_human_BT.txt")
    train_machine = os.path.join(input_folder, "train_machine.txt")
    train_machine_BT = os.path.join(input_folder,"train_machine_BT.txt")
    
    MRPC_folder = os.path.join(input_folder,"MRPC")
    if not(os.path.exists(MRPC_folder)):
        os.mkdir(MRPC_folder)
    model_folder = os.path.join(input_folder,"model")
    if not(os.path.exists(model_folder)):
        os.mkdir(model_folder)
    
    train_output = os.path.join(MRPC_folder,"train.tsv")
    dev_output = os.path.join(MRPC_folder,"dev.tsv")
    test_output = os.path.join(MRPC_folder,"test.tsv")
    
    generation_train_dev_format(train_human, train_human_BT, train_machine, train_machine_BT, train_output, True)
    generation_train_dev_format(dev_human, dev_human_BT, dev_machine, dev_machine_BT, dev_output, False)
    generation_test_format(test_human, test_human_BT, test_machine, test_machine_BT, test_output, False)
    
    
def process_classifier(process_folder, train_language, test_language):
    lines = read_all_lines(data_file)
    for idx, line in enumerate(lines):
        lines[idx] = line.strip()
    random.Random(SEED).shuffle(lines)
    
    num_samples = len(lines)
    num_train = int(num_samples * (1 - (dev_rate + test_rate)))
    num_dev = int(num_samples * dev_rate)
    train_data = lines[:num_train]
    dev_data =  lines[num_train:(num_train + num_dev)]
    test_data = lines[(num_train + num_dev):]
    
    print("train data is generating")
    generate_train_dev(process_folder, train_data, train_language, "train")
    generate_train_dev(process_folder, dev_data, train_language, "dev")
    generate_test(process_folder, test_data, train_language, test_language)
    
    generate_MRPC(process_folder)
           
def get_predictions(test_prediction):
    predictions = []
    with open(test_prediction, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        for row in reader:
            predictions.append(float(row[0]))                
    return predictions


def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]



def tokenizer(sentence):
    words = nltk.word_tokenize(sentence)
    return words;

def minimum_edit_distance(text1, text2):
    words_1 = tokenizer(text1)
    words_2 = tokenizer(text2)
    return levenshteinDistance(words_1, words_2)
        
def evaluate():
    input_folder = "./process"
    classifiers = [os.path.join(input_folder,dI) for dI in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder,dI))]
    inputs = []
    back_translations = []
    predictions = []
    classifiers.sort()
    print(classifiers)
    for idx, classifier in enumerate(classifiers):
        test_human = os.path.join(classifier, "test_human.txt")
        test_human_BT = os.path.join(classifier, "test_human_BT.txt")
        test_machine = os.path.join(classifier, "test_machine.txt")
        test_machine_BT = os.path.join(classifier, "test_machine_BT.txt")
        test_prediction = os.path.join(classifier, "test_results.tsv")
        inputs.append(read_all_lines(test_human))
        inputs[idx] += read_all_lines(test_machine)
        back_translations.append(read_all_lines(test_human_BT))
        back_translations[idx] += read_all_lines(test_machine_BT) 
        predictions.append(get_predictions(test_prediction))
    
    number_of_instances = len(predictions[0])
    number_of_classifiers = len(classifiers)
    correct_prediction = 0
    correct_classifier_prediction = 0
    
    for idx in range(number_of_instances):
        best_sim_idx = 0
        best_sim = minimum_edit_distance(inputs[0][idx], back_translations[0][idx])
        for classifier_idx in range(1, number_of_classifiers):
            current_sim = minimum_edit_distance(inputs[classifier_idx][idx], back_translations[classifier_idx][idx])
            #if current_sim > best_sim:
            if current_sim < best_sim:
                best_sim_idx = classifier_idx
                best_sim = current_sim
        
        is_machine = (idx >= (number_of_instances/2))
        prediction = predictions[best_sim_idx][idx]
        if (not(is_machine) and prediction >= 0.5) or (is_machine and prediction < 0.5):
            correct_prediction += 1
        if (is_machine and best_sim_idx == 0):
            correct_classifier_prediction += 1
    classifier_accuracy = correct_classifier_prediction/(number_of_instances/2)
    machine_detection_acc = correct_prediction / number_of_instances
    print("accuracy of classifier detection= %f"% classifier_accuracy)            
    print("accuracy of machine detection = %f"% machine_detection_acc)
    
    output_folder = "output"
    if not (os.path.exists(output_folder)):
        os.mkdir(output_folder)
    output_file = os.path.join(output_folder, "result.txt")
    
    fo = open(output_file, "w+", encoding="utf8") 
    fo.write("accuracy of classifier detection= %f\n"% classifier_accuracy)            
    fo.write("accuracy of machine detection = %f"% machine_detection_acc)
    fo.close()   


def process(train_languages, test_language):
    input_folder = "./process"
    folder_list = [os.path.join(input_folder,dI) for dI in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder,dI))]
    for sub_folder in folder_list:
        shutil.rmtree(sub_folder)

    for idx, train_language in enumerate(train_languages):
        process_folder = os.path.join('./process','classifier%d'%idx)
        if not os.path.exists(process_folder):
            os.mkdir(process_folder)
        process_classifier(process_folder, train_language, test_language)        
        #os.system(r"export GLUE_DIR=%s"%process_folder)
        #print(process_folder)
        command = r"python bert/run_classifier.py --task_name=MRPC --do_train=true --do_eval=true --data_dir=%s/MRPC --vocab_file=./bert/uncased_L-12_H-768_A-12/vocab.txt --init_checkpoint=./bert/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --bert_config_file=./bert/uncased_L-12_H-768_A-12/bert_config.json --output_dir=%s/model"%(process_folder, process_folder)        
        os.system(command)
        #print(command)
        command = r"python bert/run_classifier.py --task_name=MRPC --do_predict=true --data_dir=%s/MRPC --vocab_file=./bert/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=./bert/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=%s/model --max_seq_length=128 --output_dir=%s"%(process_folder, process_folder, process_folder)
        os.system(command)
        #print(command)
    evaluate()
train_languages = ['de','ru']
test_language = 'de'

import argparse

# initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--train_languages", "--train_languages", 
                    help="List of languages for training, separated by comma",                     
                    default = 'de')

parser.add_argument("--test_language", "--test_language", 
                    help="The language for testing",                     
                    default = 'de')

args = parser.parse_args()

train_languages = args.train_languages.split(',')
test_language = args.test_language


print(train_languages)
print(test_language)

process(train_languages, test_language)

