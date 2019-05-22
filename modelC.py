import random
import copy
import time
from queue import Queue

#HYPERPARAMS
#LENGTH_OF_POPULATION_EPOCH = 15
#POPULATION_SIZE = 10
#MUT_CONST = 3.0
#ADD_PROB = 0.3
#DELETE_PROB = 0.3
#MUTATION_PROB = 1 - ADD_PROB - DELETE_PROB
#BAD_ACCEPTER_A = 0.01
#BAD_ACCEPTER_B = 0.01

class Automaton:
    number_of_states = None
    dictionary = []
    transitions = []
    is_terminal = []

    def __init__(self, aut = None, number_of_states = 2, dictionary = []):
        if(aut is None):
            self.number_of_states = number_of_states
            self.dictionary = dictionary
            self.is_terminal = [bool(random.randint(0, 1)) for _ in range(number_of_states)]
            self.transitions = [[] for _ in range(number_of_states)]
            for i in range(number_of_states):
                for j in range(len(self.dictionary)):
                    self.transitions[i].append(random.randint(0, number_of_states - 1))
        else:
            self.number_of_states = copy.deepcopy(aut.number_of_states)
            self.dictionary = copy.deepcopy(aut.dictionary)
            self.transitions = copy.deepcopy(aut.transitions)
            self.is_terminal = copy.deepcopy(aut.is_terminal)


    def add_state(self, MUT_CONST):

        if (self.number_of_states != len(self.transitions)):
            print('PO PIZDE V ADDE0')
            print(self.number_of_states)
            print(self.transitions)

        self.number_of_states = self.number_of_states + 1
        tmp = []
        for i in range(len(self.dictionary)):
            tmp.append(random.randint(0, self.number_of_states - 1))
        self.transitions.append(tmp)
        self.is_terminal.append(bool(random.randint(0, 1)))

        n_glob = self.get_number_of_transitions()
        n = len(self.transitions)
        prob = MUT_CONST / n_glob

        for i in range(n - 1):
            for j in range(len(self.transitions[i])):
                if(random.random() < prob):
                    self.transitions[i][j] = n - 1

        if (self.number_of_states != len(self.transitions)):
            print('PO PIZDE V ADDE')
            print(self.number_of_states)
            print(self.transitions)

    def delete_state(self):
        if(self.number_of_states > 2):
            to_del = random.randint(0, self.number_of_states - 1)

            for i in range(len(self.transitions)):
                for j in range(len(self.transitions[i])):
                    if(self.transitions[i][j] == to_del):
                        self.transitions[i][j] = random.randint(0, self.number_of_states - 1)

          #  print(self.transitions)
          #  print(to_del)
            self.transitions.pop(to_del)
            self.is_terminal.pop(to_del)

            for i in range(len(self.transitions)):
                for j in range(len(self.transitions[i])):
                    if(self.transitions[i][j] >= to_del):
                        self.transitions[i][j] = self.transitions[i][j] - 1

            self.number_of_states = self.number_of_states - 1

            if (self.number_of_states != len(self.transitions)):
                print('PO PIZDE V DELETE')
                print(self.number_of_states)
                print(self.transitions)
                print(to_del)



    def check_word(self, word):
        state = 0
        cnt = 0
        while (cnt < len(word)):
            ch = word[cnt]
            nch = self.dictionary.index(ch)
            state = self.transitions[state][nch]
            cnt = cnt + 1

        return self.is_terminal[state]

    def get_accuracy(self, list_of_words):
        right, total = 0, 0
        for arr in list_of_words:
            total = total + 1
            if (self.check_word(arr[0]) == bool(int(arr[1]))):
                right = right + 1
        return float(right / total)

    def get_number_of_transitions(self):
        return len(self.dictionary) * len(self.transitions)

    def get_mutated_automaton(self, MUT_CONST):
    #    print(self.transitions)
        n_glob = self.get_number_of_transitions()
        n = len(self.transitions)
        prob = MUT_CONST / n_glob
        new_aut = Automaton(self)
        if(n < 2):
            return Automaton(aut=None, number_of_states=2, dictionary=self.dictionary)

        for i in range(n):
            for j in range(len(self.transitions[i])):
                if(random.random() < prob):
                    C = random.randint(1, n - 1)
                    new_aut.transitions[i][j] = (new_aut.transitions[i][j] + C) % n

        for i in range(n):
            if(random.random() < prob):
                new_aut.is_terminal[i] = not new_aut.is_terminal[i]

        if (n != len(self.transitions)):
            print('PO PIZDE V MUTAZII')

        return new_aut

    def minimize_automaton(self):
        used = [False for _ in self.transitions]
        q = Queue()
        q.put(0)
        used[0] = True
        while not q.empty():
            fr = q.get()
            for to in self.transitions[fr]:
                if(not used[to]):
                    used[to] = True
                    q.put(to)

        new_numbers = [-1 for _ in range(self.number_of_states)]
        cnt = 0
        for i in range(self.number_of_states):
            if (used[i]):
                new_numbers[i] = cnt
                cnt = cnt + 1


        for i in range(self.number_of_states - 1, -1, -1):
            if(used[i]):
                for j in range(len(self.transitions[i])):
                    self.transitions[i][j] = new_numbers[self.transitions[i][j]]
            else:
                self.transitions.pop(i)
                self.is_terminal.pop(i)
                self.number_of_states = self.number_of_states - 1

def generate_automaton(dataset_number, LENGTH_OF_POPULATION_EPOCH, POPULATION_SIZE, MUT_CONST, ADD_PROB, DELETE_PROB, MUTATION_PROB, BAD_ACCEPTER_A, BAD_ACCEPTER_B, TIME_LIMIT):

    start_time = time.clock()
    FILENAME = 'datasets\Block_' + str(dataset_number // 5 + 1) + '\dataset' + str(dataset_number % 5 + 1) + '.txt'
    MAX_LENGTH = 0
    DICTIONARY = []
    inputs = []
    f = open(FILENAME, 'r')

    for line in f:
        arr = line.split(',')
        for ch in arr[0]:
            if not DICTIONARY.__contains__(ch):
                DICTIONARY.append(ch)
        MAX_LENGTH = max(MAX_LENGTH, len(arr[0]))
        inputs.append(arr)

    DICTIONARY.sort()

    aut = [Automaton(number_of_states=i + 2, dictionary=DICTIONARY) for i in range(POPULATION_SIZE)]

    cnt = 0

    last_best = -1.0
    cnt_same = 0

    while (aut[0].get_accuracy(inputs) < 1.0 and time.clock() - start_time < TIME_LIMIT):
        cnt += 1
        PROB_OF_BAD_TO_BE_ACCEPTER = BAD_ACCEPTER_A + BAD_ACCEPTER_B * cnt_same
        for _ in range(LENGTH_OF_POPULATION_EPOCH):
            for i in range(POPULATION_SIZE):
                rnd_number = random.random()

                if (0 < rnd_number and rnd_number <= ADD_PROB):
                    new_aut = Automaton(aut[i])
                    new_aut.add_state(MUT_CONST)
                    if (new_aut.get_accuracy(inputs) > aut[i].get_accuracy(
                            inputs) or random.random() <= PROB_OF_BAD_TO_BE_ACCEPTER):
                        aut[i] = new_aut

                rnd_number -= ADD_PROB

                if (0 < rnd_number and rnd_number <= DELETE_PROB):
                    new_aut = Automaton(aut[i])
                    new_aut.delete_state()
                    if (new_aut.get_accuracy(inputs) > aut[i].get_accuracy(
                            inputs) or random.random() <= PROB_OF_BAD_TO_BE_ACCEPTER):
                        aut[i] = new_aut

                rnd_number -= DELETE_PROB

                if (0 < rnd_number and rnd_number <= MUTATION_PROB):
                    new_aut = aut[i].get_mutated_automaton(MUT_CONST)
                    if (new_aut.get_accuracy(inputs) > aut[i].get_accuracy(
                            inputs) or random.random() <= PROB_OF_BAD_TO_BE_ACCEPTER):
                        aut[i] = new_aut

        best_i = 0
        best_acc = aut[best_i].get_accuracy(inputs)
        for i in range(1, POPULATION_SIZE):
            acc = aut[i].get_accuracy(inputs)
            if (acc > best_acc):
                best_i = i
                best_acc = acc
        if (best_acc == last_best):
            cnt_same = cnt_same + 1
        else:
            cnt_same = 0
            last_best = best_acc
        aut[best_i].minimize_automaton()
        for i in range(0, POPULATION_SIZE):
            aut[i] = Automaton(aut[best_i])

    return aut[0].get_accuracy(inputs) == 1.0, aut[0].number_of_states, time.clock() - start_time

NUMBER_OF_DATASETS = 15
PARAMS_SET = [[15, 10, 3.0, 0.3, 0.3, 0.4, 0.1, 0.1],
              [10, 20, 5.0, 0.3, 0.4, 0.3, 0.2, 0.2],
              [50, 5, 2.0, 0.3, 0.3, 0.4, 0.01, 0.0],
              [10, 20, 7.0, 0.3, 0.4, 0.3, 0.3, 0.3],
              [10, 20, 3.0, 0.2, 0.2, 0.6, 0.1, 0.1],
              [2, 15, 3.0, 0.3, 0.3, 0.4, 0.05, 0.05],
              [15, 10, 5.0, 0.3, 0.3, 0.4, 0.0, 0.0]]

TIMELIMIT = 180
NUMBERS_OF_STATES = [4, 5, 3, 4, 4, 8, 6, 5, 10, 4, 7, 7, 10, 9, 15]
NUMBER_OF_ATTEMPTS = 10

for ii in range(len(PARAMS_SET)):
    FILENAME = 'ngalogPARAMSET=' + str(ii) + '.txt'
    print(FILENAME)
    f = open(FILENAME, 'w')

    for ds_number in range(NUMBER_OF_DATASETS):
        mean_time = 0
        min_success = 0
        success = 0
        output_message = 'D' + str(ds_number + 1) + ': '
        for i in range(NUMBER_OF_ATTEMPTS):
            is_constructed, number_of_states, time_spent = generate_automaton(ds_number, PARAMS_SET[ii][0], PARAMS_SET[ii][1], PARAMS_SET[ii][2], PARAMS_SET[ii][3], PARAMS_SET[ii][4], PARAMS_SET[ii][5], PARAMS_SET[ii][6], PARAMS_SET[ii][7], TIMELIMIT)
            if(is_constructed):
                if(number_of_states <= NUMBERS_OF_STATES[ds_number]):
                    min_success = min_success + 1
                else:
                    success = success + 1
                mean_time = mean_time + time_spent
        output_message = output_message + str(min_success) + '+' + str(success) + '/' + str(NUMBER_OF_ATTEMPTS)
        if (success + min_success > 0):
            mean_time = mean_time / (success + min_success)
            output_message = output_message + ', ' + str(float('{:.3f}'.format(mean_time))) + 's'
        print(output_message, file=f)

    f.close()


#is_constructed, nos, time = generate_automaton(0, 15, 10, 3.0, 0.3, 0.3, 0.4, 0.1, 0.1, 180)



