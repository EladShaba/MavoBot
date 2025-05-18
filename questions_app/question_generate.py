import streamlit as st
import datetime
from openai import OpenAI
import random
import re
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


if 'history' not in st.session_state:
    st.session_state.history = []
if "question_chats" not in st.session_state:
    st.session_state.question_chats = {}

def generate_prompt(question_type, topics, difficulty):
    topics_str = ", ".join(map(str, topics))
    
    base_intro = (
        f"You are ExamGenGPT, a professional CS exam question generator.\n"
        f"Create 3 questions for a university-level advanced Introduction to Computer Science course.\n"
        f"Difficulty level: {difficulty}/10.\n"
        f"Topics: {topics_str}.\n"
        f"Do not include answers or explanations — just the questions.\n"
    )

    formatting_instructions = (
        "\n\n---\n\n"
        "**Formatting Instructions:**\n"
        "- Use Markdown formatting where appropriate.\n"
        "- Use LaTeX for all mathematical notation:\n"
        "   - Inline: $E = mc^2$\n"
        "   - Block: $$a^2 + b^2 = c^2$$\n"
        "- Ensure all LaTeX expressions are valid so they render correctly in a markdown-compatible renderer.\n"
        "- Start each question block with a level 3 heading:\n"
        "   - Example: `### Question 1:`\n"
        "- Separate each question block with a horizontal rule (`---`) on its own line.\n"
    )

    if question_type == "Multiple Choice":
        question_body =(
            "Format: Provide a question followed by 4 answer options (A–D)., or a statment followed by True / false \n"
            "Example for time complexity:\n"
            "For each of the following 4 statements, indicate whether it is True or False.\n\nExample 1:\nStatement: 2^{{2n}} = O(2^n)\n\nStatement 1: ∑_{{i=1}^{{n}} i = O(n)\n\nStatement 2: There exist constants b > 0, t > 0 such that n^t = O(b^n)\n\nStatement 3: log(n!) = O(n)\n\nStatement 4: If f(n) = O(n^2), then log(f(n)) = O(log n)"
            "Example for generator:\n"
            "For each of the following generators, indicate whether it is possible or impossible to modify it to have finite delay (i.e., each next call terminates in finite time):\n\nExample 1:\nA generator producing the sequence of prime numbers.\n\nExample 2:\nA generator producing all words of a context-free grammar in CNF.\n\nExample 3:\nA generator filtering elements greater than 0 from another generator.\n\nExample 4:\nA generator producing partial sums of another generator’s sequence."
            "Example for compressions algorithms:\n"
            "For each of the following scenarios, determine whether the given statement is True or False, and justify your answer. If the statement is False, provide a counterexample.\n\nExample 1: Huffman Coding Analysis (10 points)\nGiven a frequency dictionary d = {{ 'a': 1, 'b': 3, 'c': 9, 'd': 27 }}, define d_plus_k as the dictionary with k added to each character's frequency.\n- For all k, the Huffman code length for 'a' in d equals its length in d_plus_k.\n- For all k, the Huffman code length for 'c' in d equals its length in d_plus_k.\n\nModified LZW Compression (10 points)\nFor the given Huffman code c = {{ 'a': '0', 'b': '10', 'c': '110', 'd': '1110', 'e': '1111' }}, determine the output of LZW_compress_v2 for the strings \"abcdeabccde\" and \"ededaaaaa\".\n\nThe LZW compression code is as follows:\n\ndef LZW_compress_v2(text, c, W=2**5-1, L=2**3-1):\n    intermediate = []\n    n = len(text)\n    p = 0\n    while p < n:\n        m, k = maxmatch(text, p, W, L)\n        if k <= 2:\n            if (1 + 5 + 3) >= k + len(compress(text[p:p+k], c)):\n                intermediate.append(text[p])\n                p += 1\n            else:\n                intermediate.append([m, k])\n                p += k\n    return intermediate\n\ndef inter_to_bin_v2(intermediate, c, W=2**5-1, L=2**3-1):\n    W_width = math.floor(math.log(W, 2)) + 1\n    L_width = math.floor(math.log(L, 2)) + 1\n    bits = []\n    for elem in intermediate:\n        if type(elem) == str:\n            bits.append(\"0\")\n            bits.append((bin(ord(elem))[2:]).zfill(7))\n            bits.append(c[elem])\n        else:\n            bits.append(\"1\")\n            m, k = elem\n            bits.append((bin(m)[2:]).zfill(W_width))\n            bits.append((bin(k)[2:]).zfill(L_width))\n    return bits\n\nDetermine whether the following claim is True or False:\nFor any Huffman code c and any two distinct texts, the compressed outputs are always distinct.\n\nExample 2:\nAnalyze two implementations of character counting for Huffman coding:\n\n1. For char_count(corpus), give average/worst-case time complexity\n2. For char_count2(corpus), give:\n   a) Worst-case complexity\n   b) Output for \"aaabbc\""
        )
    
    elif question_type == "Code analysis":
        question_body = (
            "Format: Provide a complex Python code snippet and ask questions about the output, time complexity, or other complex ideas.\n"
            "Example:\n"
            "Example 1:\nTask: The function find_root uses the bisection method to find a root of a continuous function f in the interval [L, U]. Determine the minimal and maximal number of iterations possible.\nInterval: L = 128.0, U = 256.0\n\ndef find_root(f, L, U, EPS=10**-10, TOL=100): \n    assert L<U \n    assert f(L)<0 and f(U)>0 \n\n    for i in range(TOL): \n        M = (L+U)/2 \n        fM = f(M) \n        print(\"Iteration\", i, \"L =\", L, \"M =\", M, \"U =\", U, \"f(M) =\", fM) \n\n if abs(fM) <= EPS: \n            print(\"Found an approximated root\") \n            return M \n        elif not L < M < U: \n            print(\"Search interval too small\") \n            return None \n        elif fM < 0: \n            L = M # continue search in upper half \n        else: # fM > 0 \n            U = M # continue search in lower half    \n\n    print(\"No root found in\", TOL, \"iterations\") \n    return None \n Example 2:Analyze the worst-case time complexity of the following recursive function f(L) in terms of O(⋅), and draw the recursion tree for a list of size n=10.\nExample 1:\nCode:\ndef f(L):\n    n = len(L)\n    if n <= 2:\n        return\n    f(L[:n//3])\n    f(L[n//3:])"
        )

    elif question_type == "Memory model":
        question_body = (
            "Format: Provide a Complex Python code snippet and ask the user to Draw the memory diagram at the end of executing the following Python code at the final state of the memory space and the namespace.\n"
            "Example 1:\nCode:\na = 1000\nb = [a, a]\ndef f(a, b):\n    a = b\n    b = b[0]\n    a[0] = 2022\n    return a\nx = f(a, b)\n\nExample 2:\nCode:\ndef update(val, arr):\n    arr[0] = val\nlst = [0]\nupdate(5, lst)"
        )

    elif question_type == "Fill in the Blanks":
        question_body = (
            "Format: code snippet with a blank (__), and ask the user to fill it.\n"
            "in order to generate a complex idea, make sure you ranomly choose a leetcode/codeforce question, then replace critical steps (or even whole chunks) of code with blanks and ask to fill it\n"
        )
    elif question_type == "Write code":
        question_body = (
       "Format: Provide a challenging coding question that is complex and aligned with the specified topics. The question can either include partial code or ask the student to implement everything from scratch.\n"
"To design a high-quality question, consider combining ideas or elements from two relevant LeetCode or Codeforces problems (based on tags or topics) to increase difficulty and depth.\n"
        "Example 1:\n"
        "Binary Search Tree – Node Size Tracking and Sorted Population\n"
        "You are given partial code for a binary search tree (BST) where each node has an additional field `size`, which is intended to store the size of the subtree rooted at that node (including the node itself).\n"
        "```python\n"
        "class Tree_node:\n"
        "    def __init__(self, key, val):\n"
        "        self.key = key\n"
        "        self.val = val\n"
        "        self.size = 0  # new field\n"
        "        self.left = None\n"
        "        self.right = None\n\n"
        "    def set_size(self):\n"
        "        def subtree_size(node):\n"
        "            if node is None:\n"
        "                return 0\n"
        "            left = subtree_size(node.left)\n"
        "            right = subtree_size(node.right)\n"
        "            return left + right + 1\n"
        "        self.size = subtree_size(self)\n\n"
        "class Binary_search_tree:\n"
        "    def set_all_sizes(self):\n"
        "        def setsize_rec(node):\n"
        "            if node is None:\n"
        "                return\n"
        "            node.set_size()\n"
        "            setsize_rec(node.left)\n"
        "            setsize_rec(node.right)\n"
        "        setsize_rec(self.root)\n"
        "```\n"
        "Tasks:\n"
        "1. Time Complexity Analysis (5 points): Analyze the worst-case time complexity of the method `set_all_sizes()`. Explain your answer briefly.\n"
        "2. Optimized Tree Size Calculation (5 points): Reimplement `set_all_sizes()` such that the worst-case runtime is O(n), where n is the number of nodes in the BST.\n"
        "3. Tree Population (10 points): Implement a method `populate_tree(lst)` that takes a sorted list `lst` and populates the existing BST (with structure already built, and all `key` fields set to None) with the values from `lst`. Do not change the structure.\n"
        "4. Complexity of Tree Population (5 points): Analyze the worst-case time complexity of `populate_tree(lst)` in terms of n.\n"
        "Example 2:\n"
        "Skyline Merge Challenge\n"
        "You are given an array of buildings, where each building is represented as [left, right, height]. A building spans from `left` to `right` on the x-axis and has the given `height`.\n"
        "Return the skyline formed by these buildings as a list of 'key points' sorted by x-coordinate. Each key point is the left endpoint of a horizontal segment in the skyline. The last point should be of height 0 to denote the end.\n"
        "Rules:\n"
        "- Merge consecutive skyline segments with the same height.\n"
        "- The skyline must change only at critical points where the height changes.\n"
        "Example Input:\n"
        "buildings = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]]\n"
        "Expected Output:\n"
        "[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]\n"
        "Write a function `get_skyline(buildings: List[List[int]]) -> List[List[int]]`.\n"
        "Example 3:\n"
        "Trapping Rain Water in Elevation Map\n"
        "You are given an array of non-negative integers representing an elevation map where the width of each bar is 1.\n"
        "Compute how much water can be trapped after it rains.\n"
        "Example Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]\n"
        "Expected Output: 6\n"
    )

    else:
        question_body = "Create a general short-answer question based on one of the topics of the course."
    return base_intro + question_body + formatting_instructions

# --- Session state init ---
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="LLM Question Generator", layout="wide")
st.title("LLM Powered Question Generator")

# --- Inject CSS for question boxes ---
st.markdown(
    """
    <style>
      .question-box {
        background-color: #f9f9f9;
        color: #111;
        padding: 1rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.5rem;
        border-left: 5px solid #4f8bf9;
      }
      .question-box h4 {
        margin-top: 0;
        color: #333;
      }
      .chat-message {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        color: #000;
        width: 80%;
      }
      .user-message {
        background-color: #e6f3ff;
        margin-left: auto;
        color: #000;
        margin-right: 0;
      }
      .assistant-message {
        background-color: #f0f0f0;
        margin-right: auto;
        color: #000;
        margin-left: 0;
      }
      .chat-container {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #eee;
      }
    </style>
    """,
    unsafe_allow_html=True
)

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
            print(content, end="", flush=True)
            full_text += content
    return full_text



# --- Top Control Panel --- 
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    question_type = st.selectbox("Select Question Type", ["Multiple Choice", "Code analysis", "Memory model", "Fill in the Blanks","Write code"])

with col2:
    difficulty = st.slider("Difficulty", 0, 10, 5)

with col3:
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.7, step=0.05)

# --- Topic Search & Tagging ---
st.markdown("#### Select Topics")

all_topics = [
    "Binary Numbers", "Floating Point Numbers",
    "Selection Sort", "Merge Sort", "Time Complexity",
    "Recursion", "Factorial", "Fibonacci", "Recursive Quicksort", "Recursive Merge Sort", "Towers of Hanoi",
    "Object Oriented Programming","List comprehensions",
    "Primality Testing", "Diffie-Hellman Key Exchange","Binary search",
    "Linked Lists", "Binary Search Trees","Two pointers","Sliding window","Matrices",
    "Generators", "Huffman Coding", "Lempel-Ziv (LZW) Compression"
]

if "topics_input" not in st.session_state:
    st.session_state.topics_input = ""

# Add a button to randomize topics
if st.button("Picks some random topics"):
    k = random.choice([2, 3])
    random_topics = random.sample(all_topics, k=k)
    st.session_state.topics_input = ", ".join(random_topics)

topics_input = st.text_input(
    label="Enter topics separated by commas",
    placeholder="Recursion, Towers of Hanoi, Recursive Quicksort, Huffman Coding, Linked List",
    key="topics_input",
    help="Example: Recursion, Towers of Hanoi, Huffman Coding"
)

selected_topics = [topic.strip() for topic in st.session_state.topics_input.split(",") if topic.strip()]

# --- Generate Button and Logic for Continuous Generation ---
st.markdown("### Generate Question")

if 'history' not in st.session_state:
    st.session_state.history = []

col_button, col_message = st.columns([1, 3])

with col_button:
    generate_new_question = st.button("Generate New Question", disabled=len(selected_topics) == 0)

with col_message:
    if not selected_topics:
        st.info("Please enter at least one topic to generate questions.")

if generate_new_question:
    with st.spinner("Generating..."):
        prompt = generate_prompt(question_type, selected_topics, difficulty)
        response = call_openai_chat(prompt, temperature)
        question_blocks = re.split(r"\n---\n", response.strip())
        question_blocks = [q.strip() for q in question_blocks if q.strip()]

        # Save to session state
        timestamp = datetime.datetime.now().isoformat()
        for q in question_blocks:
            question_id = f"{timestamp}_{len(st.session_state.history)}"
            st.session_state.history.append({
                "id": question_id,
                "question": q,
                "answer": "",
                "type": question_type,
                "topics": selected_topics,
                "timestamp": timestamp
            })
            # Initialize chat history for this question
            st.session_state.question_chats[question_id] = []


# --- Display Generated Questions ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("Generated Questions")
    # Display only the latest batch
    for i, item in enumerate(st.session_state.history):
        question_id = item.get("id", f"q_{i}")
        st.markdown(f'<div class="question-box">{item["question"]}</div>', unsafe_allow_html=True)
        # Chat interface for this question
        with st.expander("Discuss the question"):
            if question_id in st.session_state.question_chats:
                for msg in st.session_state.question_chats[question_id]:
                    if msg["role"] == "user":
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**Assistant:** {msg['content']}")
            user_question = st.text_input(
                "Ask a question about this problem:",
                key=f"question_input_{question_id}"
            )

            if st.button("Ask", key=f"ask_button_{question_id}") and user_question:
                # Ensure chat history exists
                if question_id not in st.session_state.question_chats:
                    st.session_state.question_chats[question_id] = []

                # Add user question immediately
                st.session_state.question_chats[question_id].append({
                    "role": "user",
                    "content": user_question
                })

                with st.spinner("Thinking..."):
                    prompt = f"""
                    You are ExamGenGPT, a professional CS exam question generator for a university-level advanced Introduction to Computer Science course.
                    The following is a question for the final exam generated by you:

                    {item["question"]}

                    I want you to: {user_question}


                    "**Formatting Instructions:**"
                    "- Use Markdown formatting where appropriate."
                    "- Use LaTeX for all mathematical notation:"
                    "   - Inline: $E = mc^2$"
                    "   - Block: $$a^2 + b^2 = c^2$$"
                    "- Ensure all LaTeX expressions are valid so they render correctly in a markdown-compatible renderer.\n"
                    "- Understand when to use line break and spaces"
                    "- Separate each question block with a horizontal rule (`---`) on its own line."

                    """
                    response = call_openai_chat(prompt, temperature=0.5)

                    st.session_state.question_chats[question_id].append({
                        "role": "assistant",
                        "content": response
                    })
                    st.rerun()


# --- Sidebar: Session History ---
st.sidebar.markdown("## Session History")
for item in reversed(st.session_state.history):
    with st.sidebar.expander(f"{item['timestamp'][11:19]} | {item['type']}"):
        st.markdown(f"**Topics:** {', '.join(item['topics'])}")
        st.markdown(item['question'])
        st.markdown(f"**Answer:** {item['answer']}")
