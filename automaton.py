import random

class Automaton:
    number_of_states = None
    dictionary = []
    transitions = []
    is_terminal = []

    def __init__(self, from_pk, number_of_states, dictionary, transitions, is_terminal):
        self.number_of_states = number_of_states
        self.dictionary = dictionary
        self.transitions = transitions
        self.is_terminal = is_terminal

    def add_state(self):
        self.number_of_states = self.number_of_states + 1
        tmp = []
        for i in range(len(self.dictionary)):
            tmp.append(None)
        self.transitions.append(tmp)
        self.is_terminal.append(None)

    def check_word(self, word, get_state = False):
        state = 0
        cnt = 0
        while (state is not None and cnt < len(word)):
            ch = word[cnt]
            nch = self.dictionary.index(ch)
            state = self.transitions[state][nch]
            cnt = cnt + 1

        if(get_state):
            return state
        if (state is None):
            return None
        else:
            return self.is_terminal[state]
