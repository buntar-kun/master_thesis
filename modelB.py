import prior_knowledge_miner
import automaton
import random
import copy
import time
import numpy as np

MAX_CHANGES = [1, 5, 9999999]
EPOCH_SIZES = [100, 500, 2500]

INPUTS = []
RANDOM_TESTS = 10


DOUBLED_SIZES = [[],[],[],[8, 12, 8, 12, 20, 14], [6, 10, 20, 12, 14, 20]]


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


numb_of_blocks = len(DOUBLED_SIZES)
for i_block in range(3, numb_of_blocks):
    numb_of_ds = len(DOUBLED_SIZES[i_block])
    for i_ds in range(numb_of_ds):
        if (i_block == 3 and i_ds < 5):
            i_ds = 5
        else:
            INPUTS = []
            FILENAME = "datasets\Block_" + str(i_block + 1) + "\dataset" + str(i_ds + 1) + ".txt"

            LOGFILE = "log" + str(i_block + 1) + str(i_ds + 1)
            DOUBLED_SIZE = DOUBLED_SIZES[i_block][i_ds]

            print(FILENAME)

            pkm = prior_knowledge_miner.PK_Miner(FILENAME)
            base_aut = FlexibleAutomaton(pkm.get_automaton())


            print(base_aut.transitions)

            f = open(FILENAME, 'r')
            logf = open(LOGFILE, 'w')

            for line in f:
                arr = line.split(',')
                INPUTS.append(arr)

            if not base_aut.is_correct(INPUTS):
                base_aut.get_initial_automaton()
            else:
                base_aut.fill_automaton()


            print(base_aut.transitions)

            for MAX_CHANGE in MAX_CHANGES:
                for EPOCH_SIZE in EPOCH_SIZES:
                    logf.write("MAX_CHANGE = " + str(MAX_CHANGE) + ", EPOCH_SIZE = " + str(EPOCH_SIZE) + '\n\n')
                    for t in range(RANDOM_TESTS):
                        logf.write("TEST #" + str(t) + '\n')
                        aut = copy.deepcopy(base_aut)
                        accuracy = aut.get_accuracy(INPUTS)
                        start_time = time.time()
                        while(accuracy < 1.0 and len(aut.transitions) <= DOUBLED_SIZE):
                            something_changed = False
                            for i in range(EPOCH_SIZE):
                                #generating new automaton
                                new_aut = aut.get_automaton_with_changed_connections(MAX_CHANGE)

                                new_acc = new_aut.get_accuracy(INPUTS)
                                if(accuracy < new_acc):

                                    aut = copy.deepcopy(new_aut)
                                    accuracy = new_acc
                                    something_changed = True

                            if(not something_changed):
                                aut.add_state()

                        if (accuracy < 1.0):
                            logf.write("FAIL\n")
                        else:
                            logf.write(str(len(aut.transitions)) + '\n')
                            logf.write(str(aut.transitions) + '\n')
                            logf.write(str(aut.is_terminal) + '\n')
                            logf.write(str(time.time() - start_time) + '\n\n')