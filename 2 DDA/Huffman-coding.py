import heapq

class node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ""
    
    def __lt__(self, other):
        return self.freq < other.freq

def printNodes(node, val=""):
    newVal = val + node.huff

    if node.left is None and node.right is None:
        print(f"{node.symbol} -> {newVal}")
        return

    if node.left:
        printNodes(node.left, newVal)
    if node.right:
        printNodes(node.right, newVal)

n = int(input("Enter number of characters: "))

chars = []
freqs = []

for i in range(n):
    c = input(f"Enter character {i+1}: ")
    f = int(input(f"Enter frequency of '{c}': "))
    chars.append(c)
    freqs.append(f)

# Min-heap of nodes
nodes = []
for i in range(len(chars)):
    heapq.heappush(nodes, node(freqs[i], chars[i]))

# Building Huffman Tree
while len(nodes) > 1:
    left = heapq.heappop(nodes)
    right = heapq.heappop(nodes)

    left.huff = "0"
    right.huff = "1"

    newNode = node(left.freq + right.freq, left.symbol + right.symbol, left, right)
    heapq.heappush(nodes, newNode)

print("\nHuffman Codes:")
printNodes(nodes[0])
