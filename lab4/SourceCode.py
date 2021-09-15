import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from dataset import TenseTransformDataset, DataTransformDict
import copy
import pprint

#REPRODUCIBILITY: https://pytorch.org/docs/stable/notes/randomness.html
random.seed(5) # to reproduce random.random()
torch.manual_seed(5) # to reproduce torch.randn(...), torch.randn_like(...)

class CVAE(nn.Module):
    #conditional variation auto-encoder
    class Encoder(nn.Module):
        def __init__(self, vocab_size, condition_size, condition_embedding_size, hidden_size):
            super(CVAE.Encoder, self).__init__()
            self.condition_embedding = nn.Embedding(condition_size, condition_embedding_size)
            self.char_embedding = nn.Embedding(vocab_size, hidden_size)  # parameters: the size of vocabulary, how many dimensions we want the char tensor to transfer to
            self.lstm = nn.LSTM(hidden_size, hidden_size) #parameters: the number of expected features in the input x, the number of features in the hidden state h
        def forward(self, input_char_embedded, hidden_state, cell_state):
            _, (hidden_state, cell_state) = self.lstm( input_char_embedded, (hidden_state, cell_state)) #discard encoder's output
            return _, hidden_state, cell_state

    class Decoder(nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super(CVAE.Decoder,self).__init__()
            self.condition_embedding = nn.Embedding(condition_size, condition_embedding_size)
            self.char_embedding = nn.Embedding(vocab_size, hidden_size)
            self.cellLatentCon2hidden = nn.Linear(latent_size+condition_embedding_size, hidden_size)
            self.hiddenLatentCon2hidden = nn.Linear(latent_size+condition_embedding_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size)
            self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
            self.softmax = nn.LogSoftmax(dim=-1)
        def forward(self, input_char_embedded, hidden_state, cell_state):
            output = F.relu(input_char_embedded)
            output, (hidden_state, cell_state) = self.lstm(output, (hidden_state, cell_state))
            output_char_dist = self.hidden2vocab(output)
            output_char_dist = self.softmax(output_char_dist)
            return output_char_dist, hidden_state, cell_state

    def __init__(self, vocab_size, condition_size, condition_embedding_size, hidden_size, latent_size, max_length, device):
        super(CVAE, self).__init__()
        self.vocab_size, self.condition_size, self.condition_embedding_size, self.hidden_size, self.latent_size, self.max_length, self.device = vocab_size, condition_size, condition_embedding_size, hidden_size, latent_size, max_length, device
        self.encoder = self.Encoder(vocab_size, condition_size, condition_embedding_size, hidden_size)
        self.hidden2meanLatent = nn.Linear(hidden_size, latent_size)
        self.hidden2logvarLatent = nn.Linear(hidden_size, latent_size)
        self.cell2meanLatent = nn.Linear(hidden_size, latent_size)
        self.cell2logvarLatent = nn.Linear(hidden_size, latent_size)
        self.decoder = self.Decoder(hidden_size, vocab_size)

    def forward(self, input_word, input_condition, target_condition, use_teacher_forcing):
        # ----------Encoder---------#
        # initialize hidden state and cell state and forward the encoder char by char
        hidden_state = torch.cat((self.init_hidden(), self.encoder.condition_embedding(input_condition))).view(1,1,-1)
        cell_state = torch.cat((self.init_hidden(), self.encoder.condition_embedding(input_condition))).view(1,1,-1) #hidden state and cell state share the same condition_embedding layer
        # forward the encoder char by char
        for input_char in input_word:
            input_char_embedded = self.encoder.char_embedding(input_char).view(1,1,-1) #embedding the input character and transfer the size to (sequence length, batch size, input vector's dimension)
            _, hidden_state, cell_state = self.encoder.forward(input_char_embedded, hidden_state, cell_state)
        # forward from encoder to decoder
        mean_hidden = self.hidden2meanLatent(hidden_state)
        logvar_hidden = self.hidden2logvarLatent(hidden_state)
        latent_hidden = self.reparameterize(mean_hidden, logvar_hidden)
        mean_cell = self.cell2meanLatent(cell_state)
        logvar_cell = self.cell2logvarLatent(cell_state)
        latent_cell = self.reparameterize(mean_cell, logvar_cell)

        # ----------Decoder---------#
        # handle the hidden state & cell state and forward the encoder char by char
        if target_condition == None: #training with teacher forcing, condition = input condition
            hidden_state = self.decoder.hiddenLatentCon2hidden(torch.cat((latent_hidden, self.decoder.condition_embedding(input_condition).view(1, 1, -1)), dim=-1))
            cell_state = self.decoder.cellLatentCon2hidden(torch.cat((latent_cell, self.decoder.condition_embedding(input_condition).view(1, 1, -1)), dim=-1)) #hidden state and cell state share the same condition_embedding layer
            output_word = torch.tensor([],dtype=torch.long,device=self.device)
            output_word_dist = torch.zeros(len(input_word), vocab_size, device=self.device)
            for idx_input_char, input_GT_char in enumerate(input_word):
                if input_GT_char == DataTransformDict().alpha2num['EOS']:
                    break
                if input_GT_char == DataTransformDict().alpha2num['SOS']:
                    input_char = input_GT_char #decoder's first input char must be 'SOS'
                elif use_teacher_forcing:
                    input_char = input_GT_char
                else:
                    input_char = output_char
                input_char_embedded = self.decoder.char_embedding(input_char).view(1, 1, -1)
                output_char_dist, hidden_state, cell_state = self.decoder.forward(input_char_embedded, hidden_state, cell_state)
                output_word_dist[idx_input_char+1] = output_char_dist.squeeze() # record the predicted word to compute cross entropy
                output_char = output_char_dist.argmax().view(-1) # view(-1) is to make concate available
                output_word = torch.cat((output_word.detach(),output_char.detach())) #record the predicted word with highest probability to check whether input word is same as output word
            return output_word, output_word_dist, (mean_hidden, logvar_hidden), (mean_cell, logvar_cell)

        elif target_condition != None: #testing without teacher forcing, condition = target condition
            hidden_state = self.decoder.hiddenLatentCon2hidden(torch.cat((latent_hidden, self.decoder.condition_embedding(target_condition).view(1, 1, -1)), dim=-1))
            cell_state = self.decoder.cellLatentCon2hidden(torch.cat((latent_cell, self.decoder.condition_embedding(target_condition).view(1, 1, -1)), dim=-1)) #hidden state and cell state share the same condition_embedding layer
            input_char = torch.tensor(DataTransformDict().alpha2num['SOS'],device = self.device)
            output_word = torch.tensor([], dtype=torch.long, device=self.device)
            for i in range(self.max_length):
                input_char_embedded = self.decoder.char_embedding(input_char).view(1, 1, -1)
                output_char_dist, hidden_state, cell_state = self.decoder.forward(input_char_embedded, hidden_state, cell_state)
                output_char = output_char_dist.argmax().view(-1)
                output_word = torch.cat((output_word.detach(),output_char.detach()))  # record the predicted word with highest probability to compute the BLEU-4 score
                if output_char == DataTransformDict().alpha2num['EOS']:
                    break
                input_char = output_char
            return output_word

    def generate(self, latent_hidden, latent_cell, target_condition): #generating a word with specific latent code
        # ----------Decoder---------#
        hidden_state = self.decoder.hiddenLatentCon2hidden(torch.cat((latent_hidden, self.decoder.condition_embedding(target_condition).view(1, 1, -1)), dim=-1))
        cell_state = self.decoder.cellLatentCon2hidden(torch.cat((latent_cell, self.decoder.condition_embedding(target_condition).view(1, 1, -1)),dim=-1))  # hidden state and cell state share the same condition_embedding layer
        input_char = torch.tensor(DataTransformDict().alpha2num['SOS'], device=self.device)
        output_word = torch.tensor([], dtype=torch.long, device=self.device)
        for i in range(self.max_length):
            input_char_embedded = self.decoder.char_embedding(input_char).view(1, 1, -1)
            output_char_dist, hidden_state, cell_state = self.decoder.forward(input_char_embedded, hidden_state, cell_state)
            output_char = output_char_dist.argmax().view(-1)
            output_word = torch.cat((output_word.detach(),output_char.detach()))  # record the predicted word with highest probability to compute the BLEU-4 score
            if output_char == DataTransformDict().alpha2num['EOS']:
                break
            input_char = output_char
        return output_word

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def init_hidden(self):
        return torch.zeros(self.hidden_size - self.condition_embedding_size, device=self.device)

def compute_bleu(output, reference):
    cc = SmoothingFunction()
    if len(reference) == 3:
        weights = (0.33,0.33,0.33)
    else:
        weights = (0.25,0.25,0.25,0.25)
    return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)

def compute_Gaussian_score(words):
    words_list = []
    score = 0
    yourpath = 'Lab4_dataset/train.txt'#should be your directory of train.txt
    with open(yourpath,'r') as fp:
        for line in fp:
            word = line.split(' ')
            word[3] = word[3].strip('\n')
            words_list.extend([word])
        for t in words:
            for i in words_list:
                if t == i:
                    score += 1
    return score/len(words)

def KL_weight_annealing(epoch, epoch_size, KL_annealing_mode, weight_ratio_annealing_package):
    _, _, KL_fix_ratio, KL_monotonic_ratio, KL_cyclical_ratio, KL_upper_bound = weight_ratio_annealing_package
    # KL_upper_bound determine how high the KLD_weight will be
    # KL_fix_ratio determine how long the KLD_weight will stick to 0
    if epoch / epoch_size < KL_fix_ratio:
        KLD_weight = 0
    else:
        moved_epoch = (epoch - epoch_size * KL_fix_ratio)
        moved_epoch_size = (epoch_size - epoch_size * KL_fix_ratio)
        if KL_annealing_mode == 'monotonic':
            # KL_monotonic_ratio control increasing interval's length
            if moved_epoch < KL_monotonic_ratio * moved_epoch_size:
                KLD_weight = moved_epoch * KL_upper_bound / (moved_epoch_size * KL_monotonic_ratio)
            else:
                KLD_weight = KL_upper_bound
        elif KL_annealing_mode == 'cyclical':
            # KL_cyclical_ratio control increasing interval's or 1 interval's length
            if (moved_epoch // (
                    KL_cyclical_ratio * epoch_size)) % 2 == 0:  # if the quotient is an odd number
                KLD_weight = (moved_epoch % (KL_cyclical_ratio * epoch_size)) * KL_upper_bound / (
                            epoch_size * KL_cyclical_ratio)
            else:  # if the quotient is an even number
                KLD_weight = KL_upper_bound
    return KLD_weight

def teacher_forcing_ratio_annealing(epoch, epoch_size, weight_ratio_annealing_package):
    teacher_fix_ratio, teacher_lower_bound, _, _, _, _ = weight_ratio_annealing_package
    # teacher_lower_bound determine how low the teacher forcing ratio will be
    # teacher_fix_ratio determine how long the teacher forcing ratio will stick to 1
    if epoch / epoch_size < teacher_fix_ratio:
        teacher_forcing_ratio = 1
    else:
        moved_epoch = (epoch - epoch_size * teacher_fix_ratio)
        moved_epoch_size = (epoch_size - epoch_size * teacher_fix_ratio)
        teacher_forcing_ratio = 1 - moved_epoch * (1 - teacher_lower_bound) / moved_epoch_size
    return teacher_forcing_ratio

def compute_loss(input_word, output_word_dist, KL_annealing_mode, epoch, epoch_size, mean_hidden, logvar_hidden, mean_cell, logvar_cell, weight_ratio_annealing_package):
    # KL weight annealing, to adjust how much the KL divergence should retain
    KLD_weight = KL_weight_annealing(epoch, epoch_size, KL_annealing_mode, weight_ratio_annealing_package)
    # loss function = KL weight*KLD + BCE, where BCE = nn.CrossEntropyLoss()
    BCE = nn.CrossEntropyLoss(reduction='mean')(output_word_dist[1:], input_word[1:]) #ignore 'SOS'
    KLD_hidden = -0.5 * torch.sum(1 + logvar_hidden - mean_hidden.pow(2) - logvar_hidden.exp())
    KLD_cell = -0.5 * torch.sum(1 + logvar_cell - mean_cell.pow(2) - logvar_cell.exp())
    KLD = (KLD_hidden + KLD_cell)/2
    return BCE, KLD, KLD_weight

def train_one_epoch(model, optimizer, train_dataloader, epoch, epoch_size, device, KL_annealing_mode, weight_ratio_annealing_package):
    #input should be the same as output during training phase
    model.train()
    total_L = 0
    total_BCE = 0
    total_KLD = 0
    for pair in tqdm(train_dataloader): #pair: word tensor, tense tensor
        #deal with input pair
        pair[0], pair[1] = pair[0].to(device), pair[1].to(device)
        #teacher forcing
        teacher_forcing_ratio = teacher_forcing_ratio_annealing(epoch,epoch_size, weight_ratio_annealing_package)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        #forward and compute loss
        output_word, output_word_dist, (mean_hidden, logvar_hidden), (mean_cell, logvar_cell) = model.forward(pair[0].squeeze(), pair[1].squeeze(), None, use_teacher_forcing)
        BCE, KLD, KLD_weight = compute_loss(pair[0].squeeze(), output_word_dist, KL_annealing_mode, epoch, epoch_size, mean_hidden, logvar_hidden, mean_cell, logvar_cell, weight_ratio_annealing_package)
        minibatch_L = BCE + KLD_weight*KLD
        #update
        optimizer.zero_grad()
        minibatch_L.backward()
        optimizer.step()
        total_L += minibatch_L.item()
        total_BCE += BCE.item()
        total_KLD += KLD.item()
        output_word = ''.join([DataTransformDict().num2alpha[idx.item()] for idx in output_word.detach()[:-1]])  # ignore EOS
        input_word = ''.join([DataTransformDict().num2alpha[idx.item()] for idx in pair[0].squeeze().detach()[1:-1]])  # ignore SOS and EOS
        #print(f"input word:{input_word}, output word:{output_word}")
    num_word = len(train_dataloader)
    print(f"epoch:{epoch}, KLD:{total_KLD/num_word:.5f}, BCE:{total_BCE/num_word:.5f}, loss:{total_L/num_word:.5f}, KLD_weight:{KLD_weight:.5f}, teacher forcing ratio:{teacher_forcing_ratio:.5f},last input word:{input_word},last output word:{output_word}")
    return total_KLD/num_word, total_BCE/num_word, KLD_weight, teacher_forcing_ratio

def test_one_epoch(model, test_dataloader, model_weights_path, device):
    #tense conversion
    if model_weights_path != None:  # the model has been trained well
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))  # load the model weight
    model.eval()
    total_BLEU4 = 0
    with torch.no_grad():
        for in_pair, tar_pair in test_dataloader:
            # deal with input pair and target pair
            in_pair[0], in_pair[1] = in_pair[0].squeeze().to(device), in_pair[1].squeeze().to(device)
            tar_pair[0], tar_pair[1] = tar_pair[0].squeeze().to(device), tar_pair[1].squeeze().to(device)

            # forward
            output_word = model.forward(in_pair[0], in_pair[1], tar_pair[1], None)

            # convert tensor to string
            target_word = ''.join([DataTransformDict().num2alpha[idx.item()] for idx in tar_pair[0].detach()[1:-1]])  # ignore SOS and EOS
            output_word = ''.join([DataTransformDict().num2alpha[idx.item()] for idx in output_word.detach()[:-1]])  # ignore EOS
            input_word = ''.join([DataTransformDict().num2alpha[idx.item()] for idx in in_pair[0].squeeze().detach()[1:-1]])  # ignore SOS and EOS
            total_BLEU4 += compute_bleu(output_word, target_word)
            print(f"Input: {input_word}, Target:{target_word}, Prediction:{output_word}")
    print(f"BLEU-4 score:{total_BLEU4/len(test_dataloader):.5f}")
    return total_BLEU4/len(test_dataloader)

def generate_one_epoch(model, model_weights_path, device, num_word, latent_size):
    #word generation
    if model_weights_path != None:  # the model has been trained well
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(device)))  # load the model weight
    model.eval()
    words_list = [] # record different words with different tenses
    for i in range(num_word): #generate num_word amount of words
        words = [] # record a word with different tenses
        # randomly generate latent_hidden and latent_cell from Gaussian distribution N(0,I) which decides the word's shape
        latent_hidden = torch.randn(latent_size, device=device).view(1, 1, -1)
        latent_cell = torch.randn(latent_size, device=device).view(1, 1, -1)
        # handle the specific word's tense
        for target_condition in ['sp', 'tp', 'pg', 'p']:
            target_condition = torch.tensor(DataTransformDict().tense2num[target_condition], dtype=torch.long, device=device) # convert tense from string to tensor, and reshape it to fit the LSTM's input format
            # generate word with different tenses
            output_word = model.generate(latent_hidden, latent_cell, target_condition) #generate a word according to the random latent using decoder
            output_word = ''.join([DataTransformDict().num2alpha[idx.item()] for idx in output_word.detach()[:-1]])  # ignore EOS, convert word from tensor to string
            words.append(output_word)
        words_list.append(words)
    gaussian_score = compute_Gaussian_score(words_list)
    pprint.PrettyPrinter().pprint(words_list[:10]) #pretty print
    print(f"Gaussian score:{gaussian_score}")
    return gaussian_score

def plot_results(epoch, KLD_list, BCE_list, BLEU4_list, KLD_weight_list, teacher_forcing_ratio_list, gaussian_score_list, KL_annealing_mode, weight_ratio_annealing_package, ExperimentReport_folder):
    teacher_fix_ratio, teacher_lower_bound, KL_fix_ratio, KL_monotonic_ratio, KL_cyclical_ratio, KL_upper_bound = weight_ratio_annealing_package
    with open(f"{ExperimentReport_folder}/training_messages_{KL_annealing_mode}KL.txt","w") as text_file:
        print(f"KLD: {KLD_list}", file=text_file)
        print(f"CrossEntropy: {BCE_list}", file=text_file)
        print(f"BLEU4-score: {BLEU4_list}", file=text_file)
        print(f"Gaussian-score: {gaussian_score_list}", file = text_file)
        print(f"KLD_weight(fix_ratio={KL_fix_ratio}, upper_bound={KL_upper_bound}, ratio={locals()[f'KL_{KL_annealing_mode}_ratio']}): {KLD_weight_list}", file=text_file)
        print(f"Teacher ratio(fix_ratio={teacher_fix_ratio}, lower_bound={teacher_lower_bound}): {teacher_forcing_ratio_list}", file=text_file)
    fig, ax1 = plt.subplots()
    plt.title("Training loss/ratio curve")
    plt.xlabel(f"epoch(s)")
    ax2 = ax1.twinx()
    ax1.set_ylabel("KL loss")
    ax1.plot(range(epoch+1), KLD_list, label = 'KLD', color='blue')
    ax2.plot(range(epoch+1), BCE_list, label='CrossEntropy', color='orange')
    ax2.set_ylabel("score/weight/cross entropy")
    ax2.scatter(range(epoch+1), BLEU4_list, label='BLEU4-score', s=10, color='green')
    ax2.plot(range(epoch+1), KLD_weight_list, linestyle=':', label='KLD_weight', color='red')
    ax2.plot(range(epoch+1), teacher_forcing_ratio_list, linestyle=':', label='Teacher ratio', color='purple')
    ax2.scatter(range(epoch+1), gaussian_score_list, label='Gaussian-Score', s=10, color='brown')
    fig.legend()
    plt.savefig(f"{ExperimentReport_folder}/training_process_{KL_annealing_mode}KL.png")
    plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    # ----------Construct Data Loader----------#
    train_dataset = TenseTransformDataset('./Lab4_dataset/train.txt', 'train')
    test_dataset = TenseTransformDataset('./Lab4_dataset/test.txt', 'test')
    train_dataloader = DataLoader(train_dataset, batch_size = 1)
    test_dataloader = DataLoader(test_dataset, batch_size = 1)
    #print(next(iter(train_dataloader))[0][0].squeeze())

    # ----------Hyper Parameters----------#
    epoch_size = 300
    hidden_size = 256  # RNN's hidden size is 256 or 512
    latent_size = 32
    condition_embedding_size = 8
    condition_size = len(DataTransformDict().tense2num) #the number of conditions: 4
    vocab_size = len(DataTransformDict().alpha2num)  # the number of vocabulary: 28
    empty_input_ratio = 0.1
    LR = 0.05  # learning rate
    MAX_LENGTH = 20
    num_word_generate = 100
    teacher_fix_ratio, teacher_lower_bound = 0.7, 0.7 #determine how long the teacher forcing ratio will stick to 1, determine how low the teacher forcing ratio will be
    KL_fix_ratio, KL_monotonic_ratio, KL_cyclical_ratio, KL_upper_bound = 0.2, 1, 0.15, 0.10 # determine how long the KLD_weight will stick to 0, control increasing interval's length, control increasing interval's or 1 interval's length, determine how high the KL weight will be
    weight_ratio_annealing_package = teacher_fix_ratio, teacher_lower_bound, KL_fix_ratio, KL_monotonic_ratio, KL_cyclical_ratio, KL_upper_bound
    KL_annealing_modes = ['monotonic', 'cyclical']
    mode = {'tune':False, 'demo':True} #training phase, testing phase
    resume_update = False
    ModelWeights_folder = './ModelWeights'
    ExperimentReport_folder = './ExperimentReport'

    # ----------Run Model----------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE(vocab_size, condition_size, condition_embedding_size, hidden_size, latent_size, MAX_LENGTH, device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR)
    KLD_list, BCE_list, BLEU4_list, KLD_weight_list, teacher_forcing_ratio_list, gaussian_score_list = [], [], [], [], [], []
    os.makedirs(ModelWeights_folder, exist_ok=True)
    os.makedirs(ExperimentReport_folder, exist_ok=True)
    for KL_annealing_mode in KL_annealing_modes:
        # determine to start from scratch or to consume updating
        model_weights_path = f"{ModelWeights_folder}/CVAE_{KL_annealing_mode}.pt" #determine which model weights you want to start from or which model weights you want to use in testing phase
        if os.path.isfile(model_weights_path) and resume_update:
            # if there exist an trained model weight, then resume to update it
            good_BLEU4 = test_one_epoch(model, test_dataloader, model_weights_path, device) #test one epoch will load model weights
            good_gaussian_score = generate_one_epoch(model, model_weights_path, device, num_word_generate, latent_size)
        else:
            good_BLEU4 = 0
            good_gaussian_score = 0

        # ----------Training----------#
        if mode['tune']:
            for epoch in range(epoch_size):
                #train and test
                KLD, BCE, KLD_weight, teacher_forcing_ratio = train_one_epoch(model, optimizer, train_dataloader, epoch, epoch_size, device, KL_annealing_mode, weight_ratio_annealing_package)
                BLEU4 = test_one_epoch(model, test_dataloader, None, device)
                gaussian_score = generate_one_epoch(model, None, device, num_word_generate, latent_size)

                #update the good model
                if BLEU4 > 0.5 and gaussian_score > 0.2:
                    model_weights_path = f"{ModelWeights_folder}/CVAE_{KL_annealing_mode}_BLEU{BLEU4:.5f}_G{gaussian_score:.2f}.pt" #store the model weights accord. current good BLEU4 and gaussian score
                    good_model_weights = copy.deepcopy(model.state_dict())
                    torch.save(good_model_weights, model_weights_path)
                    good_BLEU4 = BLEU4
                    good_gaussian_score = gaussian_score
                print(f"current good BLEU4 and gaussian-score over {epoch} epoch: {good_BLEU4:.5f}, {good_gaussian_score:.2f}")

                #record parameter
                KLD_list.append(KLD)
                BCE_list.append(BCE)
                BLEU4_list.append(BLEU4)
                KLD_weight_list.append(KLD_weight)
                teacher_forcing_ratio_list.append(teacher_forcing_ratio)
                gaussian_score_list.append(gaussian_score)
                plot_results(epoch, KLD_list, BCE_list, BLEU4_list, KLD_weight_list, teacher_forcing_ratio_list, gaussian_score_list, KL_annealing_mode, weight_ratio_annealing_package, ExperimentReport_folder)
            plot_results(epoch, KLD_list, BCE_list, BLEU4_list, KLD_weight_list, teacher_forcing_ratio_list, gaussian_score_list, KL_annealing_mode, weight_ratio_annealing_package, ExperimentReport_folder)
        # ----------Testing----------#
        if mode['demo']:
            #for i in range(10): #try 10 times to get the relatively high score
            #or set the random seed
            test_one_epoch(model, test_dataloader, model_weights_path, device) #model_weights_path store the best model weights during training phase
            generate_one_epoch(model, model_weights_path, device, num_word_generate, latent_size)
    print("hi")
