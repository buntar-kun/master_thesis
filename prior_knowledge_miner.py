import numpy as np
import automaton
import time

TIMEOUT = 5
MIN_OVERLAP = 10
DEBUG_OUTPUT = False

class PK_Miner:
    filename = None
    DICTIONARY = []
    MAX_LENGTH = -1
    INPUTS = []
    states = []
    is_terminal = []
    transitions = []
    dict_of_states = {}
    def __init__(self, filename):
        self.filename = filename
        self.DICTIONARY = []
        self.INPUTS = []
        self.states = []
        self.is_terminal = []
        self.transitions = []
        self.dict_of_states = []

    def first_elem(self, arr):
        return arr[0]

    def is_prefix(self, prefix, strng):
        if (len(strng) < len(prefix)):
            return False
        t = strng.find(prefix)
        return (t == 0)

    def get_words_with_prefix(self, prefix, deletePrefix=False):
        if (deletePrefix):
            n = len(prefix)
            return [[s[0][n:], s[1]] for s in self.INPUTS if self.is_prefix(prefix, s[0])]
        return [s for s in self.INPUTS if self.is_prefix(prefix, s[0])]

    def if_is_terminal(self, prefix):
        for inp in self.INPUTS:
            if (inp[0] == prefix):
                return (inp[1] == '1\n')
        return None

    def put_symbol(self, strng, char, pos):
        return strng[:(pos)] + char + strng[pos + 1:]

    def stop_criterion(self, curr_prefix, st, to):
        if (curr_prefix == '$'):
            return False

        if (time.process_time() - st > to):
            return True

        for list in self.transitions:
            for tr in list:
                if (tr == None):
                    return False
        return True

    def next_prefix(self, prefix):
        if (prefix == '$'):
            return ''
        stop_index = len(prefix) - 1
        while (stop_index != -1 and prefix[stop_index] == self.DICTIONARY[-1]):
            stop_index -= 1
        if stop_index == -1:
            return self.DICTIONARY[0] * (len(prefix) + 1)
        else:
            ans = '' + prefix
            ind = self.DICTIONARY.index(prefix[stop_index])
            ans = self.put_symbol(ans, self.DICTIONARY[ind + 1], stop_index)
            for i in range(stop_index + 1, len(prefix)):
                ans = self.put_symbol(ans, self.DICTIONARY[0], i)
            return ans

    def get_automaton(self):
        f = open(self.filename, 'r')
        for line in f:
            arr = line.split(',')
            for ch in arr[0]:
                if not self.DICTIONARY.__contains__(ch):
                    self.DICTIONARY.append(ch)
            self.MAX_LENGTH = max(self.MAX_LENGTH, len(arr[0]))
            self.INPUTS.append(arr)

        self.DICTIONARY.sort()


        self.states = []
        self.is_terminal = []
        self.transitions = []
        self.dict_of_states = {}

        self.INPUTS.sort(key=self.first_elem)
        postfixes = []
        curr_prefix = '$'
        start_time = time.process_time()
        while(not self.stop_criterion(curr_prefix, start_time, TIMEOUT)):
            curr_prefix = self.next_prefix(curr_prefix)
            if(DEBUG_OUTPUT):
                print('====================')
                print(curr_prefix)
            lst1 = self.get_words_with_prefix(curr_prefix, deletePrefix=True)
            prefix_found = False
            for k in range(len(self.states)):
                prefix = self.states[k]
                lst2 = self.get_words_with_prefix(prefix, deletePrefix=True)
                cnt = 0
                ind1, ind2 = 0, 0
                no_errors = True
                while(ind1 < len(lst1) and ind2 < len(lst2)):
                    if (lst1[ind1][0] > lst2[ind2][0]):
                        ind2 = ind2 + 1
                    elif (lst1[ind1][0] < lst2[ind2][0]):
                        ind1 = ind1 + 1
                    else:
                        if(lst1[ind1][1] != lst2[ind2][1]):
                            no_errors = False
                            if(DEBUG_OUTPUT):
                                print(prefix)
                                print(lst1[ind1])
                                print(lst2[ind2])
                                print('-------------')
                            break
                        else:
                            ind1 = ind1 + 1
                            ind2 = ind2 + 1
                            cnt += 1
                if(no_errors):
                    prefix_found = True
                    if(cnt >= MIN_OVERLAP):
                        if(DEBUG_OUTPUT):
                            print(prefix + ' is OK!!!')
                            print(cnt)

                        strng = curr_prefix[:-1]
                        chr_ind = self.DICTIONARY.index(curr_prefix[-1])
                        self.dict_of_states[curr_prefix] = self.dict_of_states[prefix]
                        if(DEBUG_OUTPUT):
                            print(strng)
                            print(self.DICTIONARY[chr_ind])
                            print(self.dict_of_states[prefix])
                            print(prefix)
                            print('--------')

                            for i in range(len(self.transitions)):
                                for j in range(len(self.transitions[i])):
                                    print(str(i) + '--' + self.DICTIONARY[j] + '-->' + str(self.transitions[i][j]))
                            print('---------------------------')
                            print(strng)
                            print(chr_ind)
                            print(self.dict_of_states.get(strng))
                        if(self.dict_of_states.get(strng) is not None and self.transitions[self.dict_of_states.get(strng)][chr_ind] is None):
                            postfixes[k].extend(lst2)
                            if(self.is_terminal[k] == None):
                                self.is_terminal[k] = self.if_is_terminal(curr_prefix)
                            self.transitions[self.dict_of_states.get(strng)][chr_ind] = self.dict_of_states[prefix]
                        break
            if(not prefix_found):
                self.states.append(curr_prefix)
                postfixes.append([lst1])
                self.transitions.append([])
                for i in range(len(self.DICTIONARY)):
                    self.transitions[-1].append(None)
                self.is_terminal.append(None)
                self.dict_of_states[curr_prefix] = len(self.states) - 1
                self.is_terminal[self.dict_of_states[curr_prefix]] = self.if_is_terminal(curr_prefix)
                if(len(self.is_terminal) > 1):
                    strng = curr_prefix[:-1]
                    chr_ind = self.DICTIONARY.index(curr_prefix[-1])
                    if(DEBUG_OUTPUT):
                        print('--NO PREFIX FOUND--')
                        print(self.dict_of_states.get(strng))
                        print(self.DICTIONARY[chr_ind])
                        print(chr_ind)
                        print(self.dict_of_states[curr_prefix])
                        print('------------')
                    self.transitions[self.dict_of_states.get(strng)][chr_ind] = self.dict_of_states[curr_prefix]
        answer = automaton.Automaton(from_pk=True, number_of_states=len(self.states), dictionary=self.DICTIONARY, transitions=self.transitions, is_terminal=self.is_terminal)
        return answer
