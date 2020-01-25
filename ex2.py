import sys
from collections import Counter
import math
"""
Igal Zaidman 311758866
Alon Gadot 305231524
"""

#read all the words form the file which are not part of the header <>
def readFile(file_name):
    content = []
    with open(file_name) as f:
        for row in f:
            if row[:5] != "<TEST" and row[:6] !="<TRAIN":
                for w in row.split():
                    content.append(w)
    f.close()
    return content

#Calculate according to Lidstone model
def calcLidstone(S,X,lamda,x_freq):
    return ((x_freq + lamda) / (S + lamda * X))

#Calculate according to HeldOut model
def calcHeldout(X,train_count,heldout_count, heldouts,word_freq):
    train_count_rev = reverseCount(train_count)
    nr_words = []
    nr = 0
    tr = 0
    
    #the number of values x that were seen r times in the training set 
    if word_freq in train_count_rev:
        nr_words = train_count_rev[word_freq]
    nr = len(nr_words)
    # special handling for unseen word
    if nr == 0:
        nr_words = set(heldouts) - set(train_count)
        nr = X - len(train_count)
    #the number of time the words that appeared r times in heldout set appeared in training set
    for w in nr_words:
        tr += heldout_count[w]
        
    return ((tr /nr) / len(heldouts))

def reverseCount(count):
    reverse = {}
    for word in count:
        if count[word] not in reverse:
            reverse[count[word]] = []
        reverse[count[word]].append(word)
    return reverse    

#Calculate the perplexity    
def calcPerplexity(model_name, S, X,train_count,valid_count,validations, lamda):
    
    valid_count_rev = {}
    p = 0
    
    #Create a set that represents the list of words from validations that appeared k times in the training for each k
    for w in validations:
        valid_count_rev[w] = train_count[w]
    valid_count_rev = reverseCount(valid_count_rev)
    
    for i,i_words in valid_count_rev.items():
        if model_name == "LID":
            X_prob = calcLidstone(S, X, lamda, i)
        #if its not Lidstone than its heldout model
        else:   
            X_prob = calcHeldout(X,train_count,valid_count,validations,i)
        p += math.log(X_prob) * sum(map(lambda word: valid_count[word], i_words))
        
    return math.exp(p / -len(validations))

if __name__ == "__main__":
    
    development_set_filename = sys.argv[1]
    test_set_filename = sys.argv[2]
    input_word = sys.argv[3]
    output_filename = sys.argv[4]
    """
    development_set_filename = "./dataset/develop.txt"
    test_set_filename = "./dataset/test.txt"
    input_word = "honduras"
    output_filename = "output.txt"
    """
    vocablury_size = 300000
    #Init
    output = []
    output.append(development_set_filename)#1 dev set file name
    output.append(test_set_filename)#2 test set file name
    output.append(input_word)#3 input word
    output.append(output_filename)#4 output file name
    output.append(vocablury_size)#5 vocablury size
    output.append(1/vocablury_size)#6 - P_uniform
    
    #Development set preprocessing
    content_dev = readFile(development_set_filename)
    S_size = len(content_dev)   
    output.append(S_size)#7 - number of events in development set
    
    #Lidstone model training
    training_size = round(S_size*0.9)
    trainings = content_dev[:training_size]
    train_count = Counter(trainings)
    validations = content_dev[training_size:]
    valid_count = Counter(validations)
    count_iw = train_count[input_word]
    count_uw = train_count["unseen-word"]
    
    output.append(S_size - training_size)#8 - num of events in validation
    output.append(training_size)#9 - num of events in training
    output.append(len(train_count))#10 - number of different events in trainig set
    output.append(count_iw)#11 - number of times input_word appears in training set
    output.append(count_iw/training_size)#12 - Pmle(input_word)
    output.append(count_uw/training_size)#13 - Pmle('unseen-word')
    lamda = 0.1
    output.append(calcLidstone(training_size,vocablury_size,lamda,count_iw))#14 - Plid(input_word)
    output.append(calcLidstone(training_size,vocablury_size,lamda,count_uw))#15 - Plid('unseen-word')
    
    #claculate perplexities for 200 different lamda's
    lamda = 0.01
    perplexities = []
    while lamda <= 2:
        perplexities.append(calcPerplexity("LID",training_size,vocablury_size,train_count,valid_count,validations, lamda))
        lamda += 0.01
    
    output.append(perplexities[0])#16 - Perplexity of validation set using lamda = 0.01
    output.append(perplexities[9])#17 - Perplexity of validation set using lamda = 0.10
    output.append(perplexities[99])#18 - Perplexity of validation set using lamda = 1.00
    opt_lamda = (perplexities.index(min(perplexities)) + 1)/100#for future calculation
    output.append(opt_lamda)#19 - The value lamda which minimizes the perplexity
    output.append(min(perplexities))#20 - The min perplexity on the validationset
    
    #held out model training
    training_size2 = round(S_size*0.5)
    trainings2 = content_dev[:training_size2]
    train_count2 = Counter(trainings2)
    heldouts = content_dev[training_size2:]
    heldout_count = Counter(heldouts)
    
    output.append(training_size2)#21 - num of events in training
    output.append(S_size - training_size2)#22 - num of events in heldout
    output.append(calcHeldout(vocablury_size,train_count2,heldout_count,heldouts,train_count2[input_word]))#23 - Pho(input_word)
    output.append(calcHeldout(vocablury_size,train_count2,heldout_count,heldouts,train_count2["unseen-word"]))#24 - Pho("unseen-word")
    
    #DEBUG of the models
    content_count = Counter(content_dev)
    unknown_size = vocablury_size - len(content_count)
    lid_check = calcLidstone(S_size,vocablury_size,opt_lamda,content_count["unseen-word"]) * unknown_size + \
        sum(map(lambda w: calcLidstone(S_size,vocablury_size,opt_lamda,content_count[w]), content_count))
    
    ho_check = calcHeldout(vocablury_size,content_count,heldout_count,heldouts,content_count["unseen-word"]) * unknown_size + \
        sum(map(lambda w: calcHeldout(vocablury_size,content_count,heldout_count,heldouts,content_count[w]), content_count))
    
    if round(lid_check,10) != 1.0 or round(ho_check,10) != 1.0:
        print("One of the models DEBUG has not passed!!!!")
    
    #Models evaluation on test set
    test_content = readFile(test_set_filename)
    test_count = Counter(test_content)
    test_size = len(test_content)
    output.append(test_size)#25 total number of events in test
    test_lid_perplexity = calcPerplexity("LID",training_size,vocablury_size,train_count,test_count,test_content, opt_lamda)
    output.append(test_lid_perplexity)#26 perplexity of test using Lidstone according to best lamda previously found
    test_ho_perplexity = calcPerplexity("HO",training_size2,vocablury_size,train_count2,test_count,test_content, opt_lamda)
    output.append(test_ho_perplexity)#27 perplexity of test using HoldOut according to best lamda previously found
    
    lid_or_ho = "H"
    if test_lid_perplexity < test_ho_perplexity:
        lid_or_ho = "L"
    output.append(lid_or_ho)#28 which model is the better one for the test data
    
    #29 - output table
    output.append("")
    output_tbl = []
    heldout_count_rev = reverseCount(train_count2)
    NTr = vocablury_size - len(train_count2) # initialize NTr to when r=0
    for r in range(10):
        f_lamda = round(calcLidstone(training_size,vocablury_size,opt_lamda,r) * training_size,5)
        f_ho = round(calcHeldout(vocablury_size,train_count2,heldout_count,heldouts,r) * training_size2,5)
        if r>0:
            NTr = len(heldout_count_rev[r])
        tr = int(f_ho * NTr)
        output_tbl.append(str(r) + "\t" + str(f_lamda) + "\t" + str(f_ho) + "\t" + str(NTr) + "\t" + str(tr))
        
    #write all the output to a file
    out = open(output_filename,"w")
    out.write("#Student\tIgal\tZaidman\tAlon\tGadot\t311758866\t305231524\n")
    #print all the outpuut line to the file
    for i,o in enumerate(output):
        out.write("#Output" + str(i+1) + "\t" + str(o) + "\n")
    #write the tables content to the file
    out.write("\n".join(output_tbl))
    out.close()
    