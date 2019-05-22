import random
import numpy as np
import copy
import time

#HYPERPARAMS (currently a generate_automaton function argument):
#mutation_prob_step = 0.01
#seqlen_before_mutation = 15

class Automaton:
    number_of_states = None
    dictionary = []
    transitions = []
    is_terminal = []

    def __init__(self, aut=None, number_of_states=2, dictionary=[]):
        if (aut is None):
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

    def randomize_state(self, data):
        if(len(data) == 1):
            self.is_terminal[data[0]] = bool(random.randint(0, 1))
        else:
            self.transitions[data[0]][data[1]] = random.randint(0, self.number_of_states - 1)

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
            if self.check_word(arr[0]) == arr[1]:
                right = right + 1
        return float(right / total)

def get_similar_automaton(aut, k, t):
    res = Automaton(aut=aut)
    number_of_tranisitions = res.number_of_states * len(res.dictionary)
    changed_trans = random.sample(range(0, number_of_tranisitions), k)
    changed_terms = random.sample(range(0, len(aut.is_terminal)), t)

    for id in changed_trans:
        fr = id // len(res.dictionary)
        ch = id % len(res.dictionary)
        while (aut.transitions[fr][ch] == res.transitions[fr][ch]):
            res.transitions[fr][ch] = random.randint(0, len(res.dictionary) - 1)

    for id in changed_terms:
        res.is_terminal[id] = not res.is_terminal[id]

    return res

def get_reputation(aut, inputs):
    t_rep = np.zeros((aut.number_of_states, len(aut.dictionary)))
    s_rep = np.zeros(aut.number_of_states)
    t_used = np.zeros((aut.number_of_states, len(aut.dictionary)))
    s_used = np.zeros(aut.number_of_states)
    for word, ans in inputs:
        sign = int(aut.check_word(word) == ans) * 2 - 1
        delta = sign / (len(word) + 1)
        state = 0
        cnt = 0
        while (cnt < len(word)):
            ch = word[cnt]
            nch = aut.dictionary.index(ch)
            if (delta < 0):
                t_rep[state][nch] = t_rep[state][nch] + delta
            t_used[state][nch] = t_used[state][nch] + 1
            state = aut.transitions[state][nch]
            cnt = cnt + 1

        if (delta < 0):
            s_rep[state] = s_rep[state] + delta
        s_used[state] = s_used[state] + 1

    not_used = []

    for i in range(aut.number_of_states):
        for j in range(len(aut.dictionary)):
            if(t_used[i][j] == 0):
                not_used.append([i, j])
            else:
                t_rep[i][j] = t_rep[i][j] / t_used[i][j]
        if(s_used[i] == 0):
            not_used.append([i])
        else:
            s_rep[i] = s_rep[i] / s_used[i]

    return t_rep, s_rep, not_used

def generate_automaton(dataset_number, mutation_prob_step, seqlen_before_mutation, time_limit, number_of_states):

    start_time = time.clock()
    FILENAME = 'datasets\Block_' + str(dataset_number // 5 + 1) + '\dataset' + str(dataset_number % 5 + 1) + '.txt'
    f = open(FILENAME, 'r')
    inputs = []
    DICTIONARY = []
    MAX_LENGTH = 0


    for line in f:
        arr = line.split(',')
        for ch in arr[0]:
            if not DICTIONARY.__contains__(ch):
                DICTIONARY.append(ch)
        MAX_LENGTH = max(MAX_LENGTH, len(arr[0]))
        inputs.append(arr)
        inputs[-1][1] = bool(int(inputs[-1][1]))

    candidate = Automaton(number_of_states=number_of_states, dictionary=DICTIONARY)

    current_accuracy = candidate.get_accuracy(inputs)

    cnt_same_acc = -seqlen_before_mutation + 1
    while(current_accuracy < 1.0 and time_limit > time.clock() - start_time):
        t_rep, s_rep, not_used = get_reputation(candidate, inputs)
        for nu in not_used:
            candidate.randomize_state(nu)
        to_change = []
        sum_mistake = 0
        for state in range(len(t_rep)):
            for ch in range(len(t_rep[state])):
                if(t_rep[state][ch] < 0):
                    to_change.append([1, state, ch, -1 * t_rep[state][ch]])
                    sum_mistake = sum_mistake + to_change[-1][-1]
        for state in range(len(s_rep)):
            if(s_rep[state] < 0):
                to_change.append([0, state, -1 * s_rep[state]])
                sum_mistake = sum_mistake + to_change[-1][-1]
        for i in range(len(to_change)):
            to_change[i][-1] = to_change[i][-1] / sum_mistake

        i_to_change = 0
        tmp = to_change[0][-1]
        rand_float = random.random()

        while(tmp < rand_float):
            i_to_change = i_to_change + 1
            tmp = tmp + to_change[i_to_change][-1]
        if(to_change[i_to_change][0] == 1):
            best, ind_best = 0, -1
            fr, ch = to_change[i_to_change][1], to_change[i_to_change][2]
            current_ind = copy.deepcopy(candidate.transitions[fr][ch])
            for to in range(number_of_states):
                candidate.transitions[fr][ch] = to
                acc = candidate.get_accuracy(inputs)
                if(acc > best):
                    best = acc
                    ind_best = to
            if(ind_best == current_ind):
                cnt_same_acc = cnt_same_acc + 1
            else:
                cnt_same_acc = -1 * seqlen_before_mutation + 1
            candidate.transitions[fr][ch] = ind_best
            current_accuracy = best
        else:
            best, ind_best = 0, -1
            st = to_change[i_to_change][1]
            current_ind = bool(candidate.is_terminal[st])
            for to in range(2):
                candidate.is_terminal[st] = bool(to)
                acc = candidate.get_accuracy(inputs)
                if(acc > best):
                    best = acc
                    ind_best = bool(to)

            if(ind_best == current_ind):
                cnt_same_acc = cnt_same_acc + 1
            else:
                cnt_same_acc = -1 * seqlen_before_mutation + 1
            candidate.is_terminal[st] = ind_best
            current_accuracy = best
        if(cnt_same_acc > 0):
            for i in range(candidate.number_of_states):
                for j in range(len(DICTIONARY)):
                    if rand_float < cnt_same_acc * mutation_prob_step:
                        candidate.randomize_state([i, j])
                if rand_float < cnt_same_acc * mutation_prob_step:
                    candidate.randomize_state([i])
            current_accuracy = candidate.get_accuracy(inputs)

    return bool(current_accuracy == 1.0), time.clock() - start_time

NUMBER_OF_DATASETS = 15
MUTATION_PROB_STEPS = [0.0, 0.01, 0.02, 0.05]
SEQLENS_BEFORE_MUTATION = [0, 15, 50]
TIMELIMIT = 180
NUMBERS_OF_STATES = [4, 5, 3, 4, 4, 8, 6, 5, 10, 4, 7, 7, 10, 9, 15]
NUMBER_OF_ATTEMPTS = 10

for mutation_prob_step in MUTATION_PROB_STEPS:
    for seqlen_before_mutation in SEQLENS_BEFORE_MUTATION:

        if(mutation_prob_step == 0.0 and SEQLENS_BEFORE_MUTATION.index(seqlen_before_mutation) != 0):
            break

        FILENAME = 'replogMPS=' + str(mutation_prob_step) + 'SBM=' + str(seqlen_before_mutation) + '.txt'
        print(FILENAME)
        f = open(FILENAME, 'w')

        for ds_number in range(NUMBER_OF_DATASETS):
            mean_time = 0
            success = 0
            output_message = 'D' + str(ds_number + 1) + ': '
            for i in range(NUMBER_OF_ATTEMPTS):
                is_constructed, time_spent = generate_automaton(ds_number, mutation_prob_step, seqlen_before_mutation, TIMELIMIT, NUMBERS_OF_STATES[ds_number])
                if(is_constructed):
                    success = success + 1
                    mean_time = mean_time + time_spent
            output_message = output_message + str(success) + '/' + str(NUMBER_OF_ATTEMPTS)
            if (success > 0):
                output_message = output_message + ', ' + str(float('{:.3f}'.format(mean_time))) + 's'
            print(output_message, file=f)

        f.close()
