import random
import copy

MAX_STRING_LENGTH = 30
DENSITY = 3
DICTIONARY_SIZE = 15
NUMBER_OF_STATES = 16
TRANSITIONS = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 15, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 15, 15, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
               [15, 15, 15, 15, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 15, 15, 15, 15, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 15, 15, 15, 15, 15, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15, 15, 15, 15, 15, 15, 15, 7, 8, 9, 10, 11, 12, 13, 14],
               [15, 15, 15, 15, 15, 15, 15, 15, 8, 9, 10, 11, 12, 13, 14], [15, 15, 15, 15, 15, 15, 15, 15, 15, 9, 10, 11, 12, 13, 14], [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 10, 11, 12, 13, 14], [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 11, 12, 13, 14],
               [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 12, 13, 14], [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 13, 14], [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 14], [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]]
TERMINALS = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]
FIRST_SYMBOL_CODE = 97
DATASET_SIZE = 15000
PERCENT_OF_POSITIVE_EXAMPLES = 0.5
STRINGS_REPEATABILITY = False


class Automata:
    size = 0
    transition = []
    isTerminal = []
    number_of_chars = 0
    def __init__(self):
        self.size = NUMBER_OF_STATES
        self.number_of_chars = DICTIONARY_SIZE
        self.transition = TRANSITIONS
        self.isTerminal = TERMINALS

    def check_word(self, word):
        current_state = 0
        for ch in word:
            current_state = self.transition[current_state][ch]
        return self.isTerminal[current_state]

    def is_reachable(self):
        if not self.transition:
            return False
        n = self.size
        is_visited = [False for i in range(n)]
        is_visited[0] = True
        q = [0]
        while(q):
            state = q.pop(0)
            for ch in range(self.number_of_chars):
                to = self.transition[state][ch]
                if(not is_visited[to]):
                    q.append(to)
                    is_visited[to] = True
        return (sum(is_visited) == n)


    def generate_string(self, length, positive = False, negative = False):
        str = [random.randint(0, self.number_of_chars - 1) for i in range(length)]
        ans = self.check_word(str)
        if (ans and negative) or (not ans and positive):
            return -1, -1
        else:
            return str, ans


def vec2word(vec):
    return ''.join(map(lambda x: chr(FIRST_SYMBOL_CODE + x), vec))

def generate_dataset(aut):
    positive = 0
    negative = 0
    last_symb = 0
    m = aut.number_of_chars
    dataset = []
    for n0 in range(DENSITY + 1):
        last_symb = m ** n0
        for x in range(last_symb):
            ans = []
            tmp = copy.copy(x)
            for s in range(n0):
                ans.append(tmp % m)
                tmp = tmp // m
            ans.reverse()
            is_positive = aut.check_word(ans)
            dataset.append([vec2word(ans), int(is_positive)])
            if(is_positive):
                positive = positive + 1
            else:
                negative = negative + 1

    remaining_pos = max(0, int(DATASET_SIZE * PERCENT_OF_POSITIVE_EXAMPLES) - positive)
    remaining_neg = max(0, int(DATASET_SIZE * (1.0 - PERCENT_OF_POSITIVE_EXAMPLES)) - negative)
    set_of_words = set()
    while(remaining_pos + remaining_neg > 0):
        print(remaining_neg)
        print(remaining_pos)
        next_str, res = aut.generate_string(random.randint(DENSITY + 1, MAX_STRING_LENGTH),
                                            positive=(remaining_neg == 0), negative=(remaining_pos == 0))
        while res == -1 or (not STRINGS_REPEATABILITY and vec2word(next_str) in set_of_words):
            next_str, res = aut.generate_string(random.randint(DENSITY + 1, MAX_STRING_LENGTH),
                                                positive=(remaining_neg == 0), negative=(remaining_pos == 0))
        set_of_words.add(vec2word(next_str))
        if(res):
            remaining_pos -= 1
        else:
            remaining_neg -= 1
        dataset.append([vec2word(next_str), int(res)])
    return dataset

aut = Automata()
print(aut.transition)
print(aut.isTerminal)
data = generate_dataset(aut)
f = open('dataset4.txt', 'w')
for d in data:
    f.write(d[0] + ',' + str(d[1]) + '\n')