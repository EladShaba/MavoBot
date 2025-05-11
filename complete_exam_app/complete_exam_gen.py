import streamlit as st
import datetime
import os
from openai import OpenAI
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



key = [
    "tf_time_complexity",
    "memory",
    "analyze",
    "few_shot_findroot",
    "writetime",
    "generator",
    "Coding question",
    "compression",
    "Explain"
]



# --- Session state init ---
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="LLM Question Generator", layout="wide")
st.title("LLM-Powered Question Generator")

def generate_exam_prompt(
    num_questions,
    pct_easy,
    pct_medium,
    pct_hard
):
    return f"""
You are ExamGenGPT, an expert in computer-science pedagogy and Python programming.
Your task is to produce a hard, high-quality final exam for an “Introduction to Computer Science” course, strictly using Python for any code examples or programming tasks.

### 1. Exam Specifications (to be supplied at runtime)
- **Number of Questions**: {num_questions}
- **Difficulty Distribution** (percentages must sum to 100):
  - Easy (Recall/Definition): {pct_easy}%
  - Medium (Application/Problem-Solving): {pct_medium}%
  - Hard (Analysis/Design): {pct_hard}%

### 2. Topic Coverage
1. **Python Basics**: Syntax, variables, conditionals, loops, lists
2. **Functions & Memory**: List comprehensions, functions, Python’s memory model
3. **Data Collections**: Tuples, sets, dictionaries, randomness
4. **Binary & Text Representation**: Binary numbers, ASCII, Unicode
5. **Floating Point & Algorithms**: Floating-point numbers, binary search
6. **Sorting & Complexity**: Selection sort, merge sort, Big-O notation
7. **Recursion**: Factorial, Fibonacci, quicksort, merge sort, Towers of Hanoi
8. **Advanced Recursion**: Memoization, exponentiation
9. **Cryptography Basics**: Primality testing, Diffie-Hellman key exchange
10. **Object-Oriented Programming**: Classes, objects
11. **Data Structures**: Linked lists, binary search trees
12. **Hashing & Hash Tables**
13. **Generators & Streams**
14. **Text Compression**: Huffman algorithm, Lempel-Ziv algorithm

### 3. Question Construction Guidelines
1. **Alignment**: Map each question clearly to one of the topics above.
2. **Clarity**: don't be short, write clearly, there is no need to write answers.
3. **Diversity**: Vary contexts (e.g. code snippets, diagrams, real-world scenarios).
4. **Python-Only**: All code tasks, examples, and snippets must be valid Python 3.

### 4. Few-Shot Examples (Optional)

Below are two exemplar exam papers. Each is

1. **Explicitly aligned** to the topics in the syllabus.  
2. **Modular**: broken into clearly labeled sections and sub-questions.  
3. **Diverse** in context—mixing code snippets, theoretical analysis, and real-world scenarios.  
4. **Python-only**: every code fragment is valid Python 3, with input/output and edge-case handling specified.

Use these as templates when crafting your own questions:  
- **Map** each sub-question to a topic.  
- **Span** multiple difficulty levels (from basic definitions to full implementations).  
- **State** expected outputs or complexity bounds unambiguously.  
- **Vary** the scenario or data structures in each section.

Important : All questions must be newly written. Do not copy any wording verbatim from the examples—instead, use them only as templates for style, structure, and topic alignment.
To design a high-quality question, consider combining ideas or elements from two relevant LeetCode or Codeforces problems (based on tags or topics) to increase difficulty and depth.
notice in examples, the first question is composed of around 4 independent sections, one of them is usually a true/false statments about time complexity and big O, another section is memory model.
instead of using the basic and well knows forms of material, you can switch to less known forms, for example instead of asking about fibonacci, we can ask about the Honsberger / Cassini's /Doubling identities. 
---
#### Example Exam 1
Question 1 (53 points)
This question consists of 5 independent sections.

Section A (5 points)
Given the following method belonging to the Binary_search_tree class:

def what(self):
    def why(n, w):
        if n is None:
            return 0
        if w % 2 == 0:
            a = why(n.left, w + 1)
            b = why(n.right, w + 1)
            return n.key + a + b
        return why(n.left, w + 1) + why(n.right, w + 1)
    return why(self.root, 0)

And given the following tree (nodes shown with key values):

          10
         /  \\
        5    15
       / \\     \\
      3   7     20

What will be returned when the function what is applied to the given tree?
Answer: ______________

Question 2 (20 points)
Given a sequence of non-negative numbers seq = [s0, s1, ..., sn-1], a subsequence of seq is a sequence sub_seq = [si0, si1, ..., sik] such that 0 ≤ i0 < i1 < ... < ik ≤ n-1.

For example, the set of subsequences of the sequence [1, 2, 3] is:
[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3].

Given a sequence of numbers seq, an increasing subsequence is one where each element is strictly greater than the one before.

Examples:
LIS([1, 2, 6, 2, 4.5, 7]) = 4
LIS([10, 5, 1]) = 1
LIS([10]) = 1
LIS([]) = 0

Section B (10 points)
Implement the function LIS_include which takes:

- lst: A list of distinct non-negative numbers with length ≥ 2.
- idxs: A list of indices (sorted ascending). The first and last indices are always included.

The function should return the length of the longest increasing subsequence of lst that includes all the numbers at the indices in idxs. Return -1 if no such subsequence exists.

Examples:
LIS_include([1, 100, 5, 200, 300, 10, 400], [0, 5, 6]) = 4
LIS_include([50, 100, 200, 300, 10, 400], [0, 3, 4, 5]) = -1

Guidelines:
Use the LIS function from the previous section.

def LIS_include(lst, idxs):
    if _____________________________________________________________:
        return -1
    res = len(idxs)
    for i in range(len(idxs)-1):
        _____________________________________________________________
        _____________________________________________________________
        _____________________________________________________________
        _____________________________________________________________
        _____________________________________________________________
        _____________________________________________________________
        _____________________________________________________________
        _____________________________________________________________
    return res

Question 3 (20 points)
This question deals with Huffman compression. Let n be the length of the corpus. The alphabet is {{"a", "b", "c", "d", "e", "f"}}.

char_count function:

def char_count(corpus):
    d = {{}}
    for ch in corpus:
        if ch in d:
            d[ch] += 1
        else:
            d[ch] = 1
    return d

Section A (4 points)
What is the time complexity of the char_count function?

Average case: ______________  
Worst case: ______________

Section B (6 points)
Modified version:

def char_count2(corpus):
    alphabet = ["a", "b", "c", "d", "e", "f"]
    char_counts = [0 for ch in alphabet]
    for ch in corpus:
        char_counts[alphabet.index(ch)] += 1
    return {{alphabet[i]: char_counts[i] for i in range(len(alphabet))}}

What is the worst-case time complexity of char_count2 in terms of n?
Answer: ______________

What will the function return for the corpus "aaabbc"?
Answer: ______________

Question 4 (25 points)
This question deals with binary search trees (BSTs) where each node has a field: key_range, a list of two integers representing the min and max key in the subtree rooted at that node.

For example, if the right child of the root has key_range [19, 38], then its subtree has keys in that range.

Tasks:
- Implement the lookup, insert, and in_subtree methods for the BinarySearchTree class.
- Ensure the key_range field is correctly maintained.
- Implement lowest_common_ancestor(node1, node2) to find the lowest common ancestor of two nodes.

Question 5 (Bonus for Groups B/C/G, 5 points)
For every constant c > 1, if f(n) = Θ(log(n)) and g(n) = Θ(n^c), then:

A. g(f(n)) = Θ((log(n))^c)  
B. There exists a constant d > 0 such that g(f(n)) = Θ(n^d)  
C. f(g(n)) = Θ((log(n))^c)  
D. f(g(n)) = Θ(log(n))

Circle the correct statement: ______________

#### Example Exam 2

Question 1 (30 points) 
This question consists of 6 independent sections. 

**True/False Statements (5 points)**  
For each of the following statements, indicate whether it is True or False by circling the correct option:

1. \(2^{{2n}} = O(2^n)\)  
2. \(\sum_{{i=1}}^n i = O(n)\)  
3. There exist constants \(b > 0, t > 0\) such that \(n^t = O(b^n)\)  
4. \(\log(n!) = O(n)\)  
5. If \(f(n) = O(n^2)\), then \(\log(f(n)) = O(\log n)\)

---

**Memory Diagram (5 points)**  
Draw the memory diagram at the end of executing the following Python code. Only show the final state of the memory space and the namespace.

python
a = 1000
b = [a, a]
def f(a, b):
    a = b
    b = b[0]
    a[0] = 2022
    return a
x = f(a, b)
1.3 Complexity Analysis (5 points):
Analyze time/space complexity of:

python
def f(L):
    n = len(L)
    if n%2==0: return max(L)
    else:
        for i in range(200, n-300):
            x = L[:i]
1.4 Bisection Method (5 points):
The function find_root uses the bisection method to find a root of a continuous function f in the interval [L, U]. Determine the minimal and maximal number of iterations possible.\nInterval: L = 128.0, U = 256.0\n\ndef find_root(f, L, U, EPS=10**-10, TOL=100): \n    assert L<U \n    assert f(L)<0 and f(U)>0 \n\n    for i in range(TOL): \n        M = (L+U)/2 \n        fM = f(M) \n        print(\"Iteration\", i, \"L =\", L, \"M =\", M, \"U =\", U, \"f(M) =\", fM) \n\n if abs(fM) <= EPS: \n            print(\"Found an approximated root\") \n            return M \n        elif not L < M < U: \n            print(\"Search interval too small\") \n            return None \n        elif fM < 0: \n            L = M # continue search in upper half \n        else: # fM > 0 \n            U = M # continue search in lower half    \n\n    print(\"No root found in\", TOL, \"iterations\") \n    return None
1.5 Recursive Complexity (5 points):
Analyze complexity and draw recursion tree for n=10:

python
def f(L):
    n = len(L)
    if n<=2: return
    f(L[:n//3])
    f(L[n//3:])
1.6 For each of the following generators, indicate whether it is possible or impossible to modify it to have finite delay (i.e., each next call terminates in finite time):\n\nExample 1:\nA generator producing the sequence of prime numbers.\n\nExample 2:\nA generator producing all words of a context-free grammar in CNF.\n\nExample 3:\nA generator filtering elements greater than 0 from another generator.\n\nExample 4:\nA generator producing partial sums of another generator’s sequence.

Question 2(25 points):
class Tree_node: \n    def __init__(self, key, val): \n        self.key = key \n        self.val = val \n        self.size = 0  # new field \n        self.left = None \n        self.right = None \n\ndef set_size(self): \n    def subtree_size(node): \n        if node == None: \n            return 0 \n        left = subtree_size(node.left) \n        right = subtree_size(node.right) \n        return left + right + 1 \n    self.size = subtree_size(self) \n\nclass Binary_search_tree: \n    def set_all_sizes(self): \n        def setsize_rec(node): \n            if node == None: \n                return \n            node.set_size() \n            setsize_rec(node.left) \n            setsize_rec(node.right) \n        setsize_rec(self.root)  \n\nGiven a binary search tree (BST) with n nodes:\n1. Analyze the worst-case time complexity of the method set_all_sizes(). Explain your answer briefly.\n2. Optimized Tree Size Calculation (5 points): Reimplement set_all_sizes() to run in O(n) time in the worst case, using recursion.\n3. Tree Population (10 points): Implement the method populate_tree(lst) for a BST where nodes initially have None keys. The method should populate the tree with keys from the sorted list lst without modifying the tree structure.\n4. Complexity of Tree Population (5 points): Analyze the worst-case time complexity of populate_tree(lst) for a BST with n nodes and a list of length n

Question 3 (20 points):
3.1 Huffman Coding (10 points):
Given d = {{'a':1, 'b':3, 'c':9, 'd':27}} and d_plus_k = d + k:
a) Is Huffman code length for 'a' always equal in d and d_plus_k?
b) Is Huffman code length for 'c' always equal in d and d_plus_k?

3.2 LZW Compression (10 points):
Given c = {{'a':'0', 'b':'10', 'c':'110', 'd':'1110', 'e':'1111'}}:
a) Show LZW_compress_v2 output for "abcdeabccde" and "ededaaaaa"
b) Is compressed output always distinct for distinct texts?

Question 4: Search Algorithms (25 points)
4.1 Exponential Search (5 points):
For t≥2, find n in [2^(t-1), 2^t) where exp_search() has maximal queries.

4.2 Guess Function (5 points):
For t≥2, find n in [2^(t-1), 2^t) where guess() has maximal queries
"""

def call_openai_chat(prompt, temperature):
    messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
    response_stream = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        temperature=temperature,
        max_output_tokens=4000,
        stream=True
    )
    
    full_text = ""
    for chunk in response_stream:
        content = getattr(chunk, "text", None)
        if content:
            print(content, end="", flush=True)  # Debug: print as it streams
            full_text += content
    return full_text

# --- Top Control Panel ---
st.title("Final Exam Generator: Introduction to CS")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    num_questions = st.number_input("Number of Questions", 1, 30, 10)

with col2:
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.7, step=0.05)

with col3:
    if st.button("✨ Inspire Me"):
        st.experimental_rerun()

# --- Difficulty Distribution ---
st.markdown("#### Difficulty Distribution (Total = 100%)")
col1, col2, col3 = st.columns(3)
with col1:
    pct_easy = st.number_input("Easy (%)", 0, 100, 30)
with col2:
    pct_medium = st.number_input("Medium (%)", 0, 100, 50)
with col3:
    pct_hard = st.number_input("Hard (%)", 0, 100, 20)



# --- Generate Button ---
if st.button("Generate Exam"):
    with st.spinner("Generating exam..."):
        prompt = generate_exam_prompt(
            num_questions, pct_easy, pct_medium, pct_hard
        )
        response = call_openai_chat(prompt, temperature)
         # Show the raw response for debugging
        st.write("Response:", response)

        st.session_state.history.append({
            "exam": response,
            "timestamp": datetime.datetime.now().isoformat()
        })

# --- Sidebar: History ---
st.sidebar.markdown("## Exam History")
for item in reversed(st.session_state.history):
    with st.sidebar.expander(f"{item['timestamp'][11:19]}"):
        st.markdown(item["exam"])

# --- Main Display ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("Latest Generated Exam")
    st.markdown(st.session_state.history[-1]["exam"])
