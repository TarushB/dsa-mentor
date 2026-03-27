"""
Build Concept Index — pre-written DSA concept notes ko FAISS mein index karta hai.

Usage:
  python scripts/build_concept_index.py
"""
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document

# Project root ko path mein add karo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TAXONOMY, CHUNK_SIZE, CHUNK_OVERLAP


# ── Pre-written Concept Notes (hamesha available, LLM ki zaroorat nahi) ──

CONCEPT_NOTES = {
    "two_pointers": """Concept: Two Pointers
When to use: Problems involving sorted arrays, pair finding, or partitioning. Look for keywords like "pair", "triplet", "sorted array", "in-place".
Core template: Initialize left=0, right=len-1. Move pointers inward based on comparison with target. For same-direction pointers, use slow/fast pattern.
Complexity: O(n) time, O(1) space
Common mistakes: Off-by-one errors with pointer boundaries. Forgetting to skip duplicates in problems like 3Sum. Not handling the case where array has fewer than 2 elements.
Related problems: Two Sum II, 3Sum, Container With Most Water, Remove Duplicates from Sorted Array, Trapping Rain Water""",

    "sliding_window": """Concept: Sliding Window
When to use: Subarray/substring problems with a contiguous constraint. Keywords: "maximum sum subarray of size k", "longest substring", "minimum window".
Core template: left pointer, right pointer, expand right, shrink left when constraint violated. Track window state in a hash map or counter.
Complexity: O(n) time, O(1) or O(k) space where k is the window/charset size
Common mistakes: Forgetting to shrink window when constraint violated. Not updating the answer at the right time (before or after shrinking). Confusing fixed-size vs variable-size window patterns.
Related problems: Longest Substring Without Repeating Characters, Minimum Window Substring, Maximum Sum Subarray of Size K, Sliding Window Maximum""",

    "binary_search": """Concept: Binary Search
When to use: Sorted arrays, monotonic functions, search space reduction. Keywords: "sorted", "find minimum", "search", "rotated sorted array".
Core template: lo, hi = 0, len-1. While lo <= hi: mid = (lo+hi)//2. Compare and adjust lo or hi. For "find first/last" variants, use lo < hi with careful boundary adjustment.
Complexity: O(log n) time, O(1) space
Common mistakes: Infinite loops from wrong boundary updates (lo=mid vs lo=mid+1). Wrong condition for upper/lower bound variants. Not handling empty arrays.
Related problems: Search in Rotated Sorted Array, Find First and Last Position, Search a 2D Matrix, Koko Eating Bananas""",

    "prefix_sum": """Concept: Prefix Sum
When to use: Range sum queries, subarray sum problems. Keywords: "sum of subarray", "range sum", "cumulative".
Core template: Build prefix array where prefix[i] = sum(arr[0:i]). Range sum = prefix[right+1] - prefix[left]. For 2D, use inclusion-exclusion.
Complexity: O(n) build time, O(1) per query, O(n) space
Common mistakes: Off-by-one in prefix array indexing. Forgetting the empty prefix (prefix[0]=0). Not considering negative numbers in subarray sum problems.
Related problems: Product of Array Except Self, Subarray Sum Equals K, Range Sum Query, Contiguous Array""",

    "monotonic_stack": """Concept: Monotonic Stack
When to use: Next greater/smaller element problems. Keywords: "next greater", "previous smaller", "stock span", "histogram".
Core template: Iterate through array. While stack is non-empty and current element violates monotonic property, pop and process. Push current element.
Complexity: O(n) time, O(n) space
Common mistakes: Confusing increasing vs decreasing stack direction. Not handling remaining elements in stack after iteration. Incorrect comparison operator (< vs <=).
Related problems: Next Greater Element, Daily Temperatures, Largest Rectangle in Histogram, Sliding Window Maximum""",

    "bfs": """Concept: Breadth-First Search (BFS)
When to use: Shortest path in unweighted graphs, level-order traversal, minimum steps problems. Keywords: "shortest path", "minimum moves", "level order".
Core template: Initialize queue with start. While queue: pop front, process, add unvisited neighbors. Track visited set. For shortest path, track distance.
Complexity: O(V + E) time, O(V) space
Common mistakes: Forgetting visited set (infinite loops). Not checking visited before adding to queue. Processing node when popping vs when adding.
Related problems: Binary Tree Level Order Traversal, Number of Islands, Word Ladder, Shortest Path in Binary Matrix""",

    "dfs": """Concept: Depth-First Search (DFS)
When to use: Graph traversal, tree problems, path finding, connected components. Keywords: "all paths", "connected components", "tree traversal".
Core template: Recursive: mark visited, process, recurse on neighbors. Iterative: use stack instead of queue. For trees, no visited set needed.
Complexity: O(V + E) time, O(V) space (recursion stack)
Common mistakes: Stack overflow on deep recursion. Forgetting to mark visited before recursing. Not handling disconnected components.
Related problems: Number of Islands, Clone Graph, Path Sum, Validate BST, Course Schedule""",

    "backtracking": """Concept: Backtracking
When to use: Generate all combinations/permutations, constraint satisfaction. Keywords: "all possible", "generate", "combinations", "permutations", "N-Queens".
Core template: Define choices at each step. Make a choice, recurse, undo the choice (backtrack). Prune invalid branches early.
Complexity: Often exponential O(2^n) or O(n!), but pruning helps in practice
Common mistakes: Not undoing the choice (modifying state permanently). Missing base case. Not pruning early enough, leading to TLE.
Related problems: Subsets, Permutations, N-Queens, Combination Sum, Word Search""",

    "dp_1d": """Concept: 1D Dynamic Programming
When to use: Optimization problems with overlapping subproblems and optimal substructure. Keywords: "minimum cost", "number of ways", "maximum profit", "can you reach".
Core template: Define dp[i] meaning. Find recurrence: dp[i] = f(dp[i-1], dp[i-2], ...). Initialize base cases. Fill table bottom-up or use memoization top-down.
Complexity: O(n) time, O(n) space (often reducible to O(1) with rolling variables)
Common mistakes: Wrong recurrence relation. Missing base cases. Not considering all transitions. Index out of bounds on dp array.
Related problems: Climbing Stairs, House Robber, Coin Change, Longest Increasing Subsequence, Word Break""",

    "dp_2d": """Concept: 2D Dynamic Programming
When to use: Problems involving two sequences or a grid. Keywords: "edit distance", "longest common subsequence", "grid paths", "matrix".
Core template: dp[i][j] represents solution for first i elements of A and first j elements of B. Recurrence considers dp[i-1][j], dp[i][j-1], dp[i-1][j-1].
Complexity: O(m*n) time, O(m*n) space (sometimes reducible to O(min(m,n)))
Common mistakes: Incorrect dimensions or indexing. Forgetting 0-indexed vs 1-indexed conversion. Not initializing first row and column correctly.
Related problems: Longest Common Subsequence, Edit Distance, Unique Paths, Minimum Path Sum, Interleaving String""",

    "dp_interval": """Concept: Interval DP
When to use: Problems where you merge/split intervals or ranges. Keywords: "merge stones", "burst balloons", "matrix chain multiplication".
Core template: dp[i][j] = optimal answer for range [i, j]. Iterate by interval length. For each interval, try all split points k: dp[i][j] = min/max(dp[i][k] + dp[k+1][j] + cost).
Complexity: O(n^3) time, O(n^2) space
Common mistakes: Wrong iteration order (must process short intervals before long ones). Not considering all split points. Incorrect cost function.
Related problems: Burst Balloons, Minimum Cost to Merge Stones, Matrix Chain Multiplication, Palindrome Partitioning II""",

    "greedy": """Concept: Greedy Algorithm
When to use: Problems where local optimal choices lead to global optimal. Keywords: "minimum number of", "maximum number of", "schedule", "interval".
Core template: Sort by some criterion. Iterate and make the locally optimal choice at each step. Prove correctness by exchange argument.
Complexity: Usually O(n log n) due to sorting, O(n) for the greedy pass
Common mistakes: Assuming greedy works without proof (many problems require DP instead). Wrong sorting criterion. Not handling ties correctly.
Related problems: Merge Intervals, Non-overlapping Intervals, Jump Game, Task Scheduler, Gas Station""",

    "union_find": """Concept: Union-Find (Disjoint Set Union)
When to use: Dynamic connectivity, grouping, cycle detection in undirected graphs. Keywords: "connected components", "group", "union", "merge".
Core template: parent[] array with find(x) using path compression and union(x,y) with rank/size. find() returns root, union() merges two sets.
Complexity: Nearly O(1) amortized per operation with path compression + union by rank
Common mistakes: Forgetting path compression (degrades to O(n)). Not using union by rank. Off-by-one in component counting.
Related problems: Number of Connected Components, Redundant Connection, Accounts Merge, Number of Islands""",

    "trie": """Concept: Trie (Prefix Tree)
When to use: Prefix-based search, autocomplete, word dictionary operations. Keywords: "prefix", "word search", "starts with", "dictionary".
Core template: TrieNode with children dict and is_end flag. Insert: walk and create nodes. Search: walk and check is_end. StartsWith: walk without checking is_end.
Complexity: O(m) per operation where m is word length, O(N*m) space for N words
Common mistakes: Not marking end of word correctly. Memory-inefficient implementation. Not handling the empty string case.
Related problems: Implement Trie, Word Search II, Design Add and Search Words, Replace Words""",

    "heap_kth_element": """Concept: Heap / Priority Queue / Kth Element
When to use: Finding kth largest/smallest, merge k sorted lists, streaming median. Keywords: "kth largest", "top k", "merge sorted", "median".
Core template: For kth largest: maintain min-heap of size k. For kth smallest: maintain max-heap of size k. For merge k sorted: min-heap of (value, list_index, element_index).
Complexity: O(n log k) for kth element, O(n log n) for full heap sort
Common mistakes: Min-heap vs max-heap confusion. Not maintaining heap size correctly. Forgetting Python's heapq is min-heap only (negate for max-heap).
Related problems: Kth Largest Element, Top K Frequent Elements, Merge K Sorted Lists, Find Median from Data Stream""",

    "graph_topological": """Concept: Topological Sort
When to use: Dependency ordering, course scheduling, build systems. Keywords: "prerequisites", "order", "dependency", "schedule", "DAG".
Core template: Kahn's algorithm (BFS): compute in-degrees, start from 0-degree nodes, remove edges, repeat. DFS: post-order traversal then reverse.
Complexity: O(V + E) time, O(V) space
Common mistakes: Not detecting cycles (topological sort only works on DAGs). Not handling disconnected components. Wrong in-degree initialization.
Related problems: Course Schedule, Course Schedule II, Alien Dictionary, Minimum Height Trees""",

    "bit_manipulation": """Concept: Bit Manipulation
When to use: Problems involving binary representations, XOR properties, power of 2 checks. Keywords: "single number", "power of two", "binary", "XOR".
Core template: XOR to find unique: a^a=0, a^0=a. Check bit: n & (1<<i). Set bit: n | (1<<i). Clear bit: n & ~(1<<i). Check power of 2: n & (n-1) == 0.
Complexity: O(1) or O(log n) per operation
Common mistakes: Operator precedence (& has lower precedence than ==). Signed vs unsigned integer issues. Not handling negative numbers.
Related problems: Single Number, Number of 1 Bits, Counting Bits, Power of Two, Reverse Bits""",

    "math": """Concept: Math-Based Problems
When to use: Number theory, modular arithmetic, geometry. Keywords: "prime", "GCD", "factorial", "modulo", "palindrome number".
Core template: GCD: Euclidean algorithm. Modular arithmetic: (a*b) % m = ((a%m)*(b%m)) % m. Sieve of Eratosthenes for primes up to N.
Complexity: Varies; GCD is O(log min(a,b)), Sieve is O(n log log n)
Common mistakes: Integer overflow in intermediate calculations. Negative modulo edge cases. Floating point precision issues in geometry.
Related problems: Climbing Stairs, Pow(x,n), Count Primes, Happy Number, Excel Sheet Column Number""",
}


# ── Chunking — bade text ko chote pieces mein todo ───────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Text ko overlapping chunks mein split karo approximate token count (word-based) se."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ── Main Builder — yahan se index banta hai ───────────────────────

def build_concept_index() -> int:
    """
    Pre-written concept notes se FAISS concept index banao.

    Returns:
        Kitne documents index hue
    """
    from rag.embeddings import create_index

    documents = []

    for pattern in TAXONOMY:
        display_name = pattern.replace("_", " ").title()
        print(f"  Processing: {display_name}...", end=" ")

        concept_text = CONCEPT_NOTES.get(pattern, f"Concept: {display_name}\nNo detailed notes available.")

        # Chunk karo aur documents banao
        chunks = chunk_text(concept_text)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "pattern": pattern,
                    "concept_name": display_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": "pre_written",
                },
            )
            documents.append(doc)

        print(f"→ {len(chunks)} chunk(s)")

    # FAISS index banao
    print(f"\n🔨 Building FAISS concept index with {len(documents)} documents...")
    create_index(documents, "concepts")
    print("Concept index saved.")

    return len(documents)


def main():
    print("\nBuilding DSA Concept Knowledge Base\n")
    count = build_concept_index()
    print(f"\nDone! Indexed {count} concept documents across {len(TAXONOMY)} patterns.")


if __name__ == "__main__":
    main()
