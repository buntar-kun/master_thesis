import prior_knowledge_miner
import automaton
import random
import copy
import time
import numpy as np
import random
import numpy as np


DICTIONARY = ['a', 'b']
INPUTS = []
NUMBER_OF_POSITIONS = 15
block = 3
ds = 5

START_POSITION = np.zeros(NUMBER_OF_POSITIONS)
START_POSITION[0] = 1
NUMBER_OF_CHARS = len(DICTIONARY)
PERCENT_OF_TESTS = 0.1
TenzToAdd = 2.0
EPS = 0.0001
NU = 0.7
NU_ADDER = 0.3

class NeuralNetwork:

    tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    adder = np.zeros(NUMBER_OF_POSITIONS)
    prior_data = []
    terminals = []

    def __init__(self, prior_data = [], terminals = []):
        self.tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
        self.prior_data = prior_data
        self.terminals = terminals
        for fr in range(NUMBER_OF_POSITIONS):
            for ch in range(NUMBER_OF_CHARS):
                z = np.random.rand(NUMBER_OF_POSITIONS)
                z = normalize(z)
                for to in range(NUMBER_OF_POSITIONS):
                    self.tensor[to][ch][fr] = z[to]
        self.adder = np.zeros(NUMBER_OF_POSITIONS)
        for to in range(NUMBER_OF_POSITIONS):
            self.adder[to] = random.random()
        self.remind_prior_knowledge()

    def remind_prior_knowledge(self, pr=False):
        fr = -1
        for data in self.prior_data:
            fr = fr + 1
            for ch in range(len(DICTIONARY)):
                if(data[ch] != None):
                    for to in range(NUMBER_OF_POSITIONS):
                        if(data[ch] == to):
                            self.tensor[to][ch][fr] = 1.0
                        else:
                            self.tensor[to][ch][fr] = 0.0
        for i in range(len(self.terminals)):
            self.adder[i] = float(self.terminals[i])

    def check(self, word):
        curr_pos = START_POSITION
        for k in range(len(word)):
            curr_word = char_to_vector(word[k])
            curr_pos = match(self, curr_word, curr_pos)
            curr_pos = normalize(curr_pos)
        return lastsum(self, curr_pos)

    def train_online(self, dataset, start_time):
        average_error = 1.0
        epoch_number = 0
        n = len(dataset)
        tests_size = int(PERCENT_OF_TESTS * n)
        while (average_error > EPS and time.clock() - start_time < 180):
            random.shuffle(dataset)
            cases_left = len(dataset)
            epoch_number += 1
      #      print ('Epoch #' + str(epoch_number))
            while(cases_left > tests_size):
                self.train(dataset[cases_left - 1][0], dataset[cases_left - 1][1])
                cases_left -= 1
            average_error = 0.0
            for i in range(cases_left):
                average_error += cost_function(dataset[i][1], self.check(dataset[i][0]))
            average_error /= cases_left
      #      print ("Average error: " + str(average_error))

        if(time.clock() - start_time< 60):
            return True
        else:
            return False

    def train(self, word, exp):
        cut_v = np.vectorize(cut)
        word_length = len(word)
        positions = np.zeros([word_length + 1, NUMBER_OF_POSITIONS])
        before_normalize = np.zeros([word_length, NUMBER_OF_POSITIONS])
        d_tensor = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
        d_adder = np.zeros(NUMBER_OF_POSITIONS)
        positions[0] = START_POSITION

        for k in range(word_length):
            curr_word = char_to_vector(word[k])
            before_normalize[k] = match(self, curr_word, positions[k])
            positions[k + 1] = normalize(before_normalize[k])
        answer = lastsum(self, positions[-1])
        error = cost_function(exp, answer)


        gradient = cost_function_derrivative(exp, answer)
        d_adder += lastsum_derrivative_adder(self, gradient, positions[-1])
        gradient = lastsum_derrivative(self, gradient)
        first_gradient = sum(abs(gradient)) * TenzToAdd
        for k in range(word_length - 1, -1, -1):
            curr_grad = sum(abs(gradient))
            if (curr_grad < 0.001):
                koef = 1.0
            else:
                koef = first_gradient / sum(abs(gradient))
            gradient *= koef
            curr_word = char_to_vector(word[k])
            gradient = normalize_derrivative(gradient, before_normalize[k])
            d_tensor += match_derrivative_tensor(gradient, curr_word, positions[k])
            gradient = match_derrivative(self, gradient, curr_word)

        d_tensor /= max(1, word_length)
        self.tensor = cut_v(self.tensor - NU * d_tensor)
        self.adder = cut_v(self.adder - NU * NU_ADDER * d_adder)
        self.remind_prior_knowledge()
        return error

    def get_automaton(self):
        for j in range(NUMBER_OF_POSITIONS):
            for i in range(NUMBER_OF_CHARS):
                max_ind = 0
                for k in range(1, NUMBER_OF_POSITIONS):
                    if (nn.tensor[k][i][j] > nn.tensor[max_ind][i][j]):
                        max_ind = k
                print (str(j) + "--" + str(DICTIONARY[i]) + '-->' + str(max_ind))
        for k in range(NUMBER_OF_POSITIONS):
            if (nn.adder[k] > 0.5):
                print (str(k) + " is terminal")

def cost_function(exp, res):
    return (res - exp) ** 2

def cost_function_derrivative(exp, res):
    return res - exp

def match(nn, ch, pos):
    new_pos = np.zeros(NUMBER_OF_POSITIONS)
    for k in range(NUMBER_OF_POSITIONS):
        for i in range (NUMBER_OF_CHARS):
            for j in range (NUMBER_OF_POSITIONS):
                new_pos[k] += nn.tensor[k][i][j] * ch[i] * pos[j]
    return new_pos

def match_derrivative(nn, dz, ch):
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_POSITIONS):
        for k in range(NUMBER_OF_POSITIONS):
            for j in range (NUMBER_OF_CHARS):
                derrivative[k][i] += nn.tensor[k][j][i] * ch[j]
    return np.dot(dz, derrivative)


def match_derrivative_tensor(dz, ch, pos):
    sample_matrix = np.zeros([NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_CHARS):
        for j in range(NUMBER_OF_POSITIONS):
            sample_matrix[i][j] = ch[i] * pos[j]
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_CHARS, NUMBER_OF_POSITIONS])
    for i in range(NUMBER_OF_POSITIONS):
        derrivative[i] = dz[i] * sample_matrix
    return derrivative

def normalize(t):
    return t / np.sum(t)

def normalize_derrivative(dz, inp):
    derrivative = np.zeros([NUMBER_OF_POSITIONS, NUMBER_OF_POSITIONS])
    sum = np.sum(inp)
    for i in range(NUMBER_OF_POSITIONS):
        for j in range(NUMBER_OF_POSITIONS):
            if(i == j):
                derrivative[i][j] = (sum - inp[i]) / (sum ** 2)
            else:
                derrivative[i][j] = -inp[i] / (sum ** 2)
    return np.dot(dz, derrivative)

def lastsum(nn, x):
    return np.dot(nn.adder, x)

def lastsum_derrivative(nn, dz):
    return np.dot(dz, nn.adder)

def lastsum_derrivative_adder(nn, dz, inp):
    derrivative = np.multiply(inp, nn.adder)
    return np.dot(dz, derrivative)

def char_to_vector(ch):
    index = DICTIONARY.index(ch)
    vec = np.zeros(NUMBER_OF_CHARS)
    vec[index] = 1.0
    return vec

def cut(x):
    if (x > 1.0):
        return 1.0
    if (x < 0.0):
        return 0.0
    return x

class FlexibleAutomaton(automaton.Automaton):
    untrusted_data = []
    def __init__(self, aut, from_FA=False):
        self.dictionary = aut.dictionary
        self.number_of_states = aut.number_of_states
        self.is_terminal = aut.is_terminal
        self.transitions = aut.transitions

        #complete automata to fullness
        if from_FA:
            self.untrusted_data = aut.untrusted_data

    def fill_automaton(self):
        for i in range(self.number_of_states):
            for j in range(len(self.dictionary)):
                if(self.transitions[i][j] == None):
                    self.transitions[i][j] = random.randint(0, self.number_of_states - 1)
                    self.untrusted_data.append([i, j])
        for i in range(self.number_of_states):
            if(self.is_terminal[i] == None):
                self.is_terminal[i] = bool(random.randint(0, 1))
                self.untrusted_data.append([-1, i])

    def get_initial_automaton(self):
        self.number_of_states = 2
        self.is_terminal = [False, True]
        self.untrusted_data = [[-1, 0], [-1, 1]]
        self.transitions = [[], []]
        for i in range(len(self.dictionary)):
            self.transitions[0].append(1)
            self.transitions[1].append(0)
            self.untrusted_data.append([0, i])
            self.untrusted_data.append([1, i])


    def add_state(self):
        self.number_of_states = self.number_of_states + 1
        tmp = []
        for i in range(len(self.dictionary)):
            tmp.append(random.randint(0, self.number_of_states - 1))
            self.untrusted_data.append([self.number_of_states - 1, i])
        self.transitions.append(tmp)
        self.is_terminal.append(bool(random.randint(0, 1)))
        self.untrusted_data.append([-1, self.number_of_states - 1])

    def get_automaton_with_changed_connections(self, MAX_CHANGE):
        mc = min(MAX_CHANGE, len(self.untrusted_data))
        ans = FlexibleAutomaton(copy.deepcopy(self), from_FA=True)
        number_of_changes = random.randint(1, mc)
        changes = np.random.choice(len(self.untrusted_data), size=number_of_changes, replace=False)
        for i in changes:
            tr = ans.untrusted_data[i]
            if (tr[0] == -1):
                ans.is_terminal[tr[1]] = not ans.is_terminal[tr[1]]
            else:
                ans.transitions[tr[0]][tr[1]] = random.randint(0, ans.number_of_states - 1)

        return ans

    def check_word(self, word, result):
        state = 0
        for ch in word:
            nch = self.dictionary.index(ch)
            state = self.transitions[state][nch]
        return (self.is_terminal[state] == result)

    def get_accuracy(self, list_of_words):
        right, total = 0, 0
        for arr in list_of_words:
            total = total + 1
            if self.check_word(arr[0], bool(int(arr[1]))):
                right = right + 1
        return float(right / total)

    def is_correct(self, inputs):
        mistake_found = False
        for inp in inputs:
            state = 0
            for ch in inp[0]:
                if state is not None:
                    nch = self.dictionary.index(ch)
                    state = self.transitions[state][nch]
            mistake_found = mistake_found or not (state is None or self.is_terminal[state] == bool(int(inp[1])))

        return not mistake_found


cnt = 0
sum_time = 0

for ii in range (10):
    start = time.clock()
    INPUTS = []
    FILENAME = "datasets\Block_" + str(block) + "\dataset" + str(ds) + ".txt"

    pkm = prior_knowledge_miner.PK_Miner(FILENAME)
    base_aut = FlexibleAutomaton(pkm.get_automaton())


   # print(base_aut.transitions)

    f = open(FILENAME, 'r')
    dataset = []

    for line in f:
        arr = line.split(',')
        INPUTS.append(arr)
        isOk = 1.0
        if arr[1][0] == '0':
            isOk = 0.0
        dataset.append([arr[0], isOk])


    nn = NeuralNetwork(prior_data=pkm.transitions, terminals=pkm.is_terminal);
    success = nn.train_online(dataset, start)
    if(success):
        cnt = cnt + 1
        sum_time += time.clock() - start
    nn.get_automaton()
    print(str(ii) + " " + str(time.clock() - start))

print("Successes: " + str(cnt))
if(cnt > 0):
    print("Average time: " + str(sum_time / cnt))