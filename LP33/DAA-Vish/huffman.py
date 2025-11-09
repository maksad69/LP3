import heapq

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq, self.symbol, self.left, self.right, self.huff = freq, symbol, left, right, ''
    def __lt__(self, nxt): return self.freq < nxt.freq

def print_nodes(node, val=''):
    code = val + str(node.huff)
    if node.left: print_nodes(node.left, code)
    if node.right: print_nodes(node.right, code)
    if not node.left and not node.right: print(f"{node.symbol} -> {code}")

chars, freq = ['a','b','c','d','e','f'], [5,9,12,13,16,45]
nodes = [Node(freq[i], chars[i]) for i in range(len(chars))]
heapq.heapify(nodes)

while len(nodes) > 1:
    l, r = heapq.heappop(nodes), heapq.heappop(nodes)
    l.huff, r.huff = '0', '1'
    heapq.heappush(nodes, Node(l.freq + r.freq, l.symbol + r.symbol, l, r))

print_nodes(nodes[0])
