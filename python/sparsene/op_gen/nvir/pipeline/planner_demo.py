from itertools import permutations, product

def is_valid_permutation(perm, constraints):
    """
    Check if a permutation satisfies the constraints.
    Each constraint is a pair (m, n) meaning m must not appear after n.
    """
    for m, n in constraints:
        if perm.index(m) > perm.index(n):  # If m appears after n, it's invalid
            return False
    return True

def generate_all_partitions_with_constraints(m, constraints):
    # Generate all permutations of the elements 0, 1, ..., m-1
    elements = list(range(m))
    all_permutations = permutations(elements)

    # Filter permutations based on the constraints
    valid_permutations = filter(lambda perm: is_valid_permutation(perm, constraints), all_permutations)

    # For each valid permutation, generate all possible ways to insert delimiters
    results = []
    for perm in valid_permutations:
        # There are m-1 gaps between m elements, and each gap can either have a delimiter or not
        num_gaps = m - 1
        for delimiter_plan in product([0, 1], repeat=num_gaps):
            partition = []
            current_container = []
            for i, elem in enumerate(perm):
                current_container.append(elem)
                # If there's a delimiter in this gap, start a new container
                if i < num_gaps and delimiter_plan[i] == 1:
                    partition.append(current_container)
                    current_container = []
            # Add the last container
            partition.append(current_container)
            results.append(partition)
    
    return results

# 1. 我会对op做排列 -> m!
# 2. 根据opgraph的约束，剪枝掉不合法的排列 -> 得到合法的排列
# 3. 生成分隔符，插入到排列之间 -> 2^(m-1) (max_stage=4) (min_op_score_sum_in_stage=2)
# 4. 遍历可能的shifts(range=1~3)

# Example usage
m = 9  # Number of elements
constraints = [
	(0, 1),
    (0, 2),
    (1, 3),
	(2, 3),
	(3, 4),
    (5, 6),
    (6, 7),
    (4, 8),
    (7, 8),
]

partitions = generate_all_partitions_with_constraints(m, constraints)

# Print the results
for p in partitions:
    print(p)

print(f"Total number of partitions: {len(partitions)}")

import math

def calculate_T(m):
    """
    Calculate the total number of possibilities T(m) for arranging m elements 
    into arbitrary numbers of ordered containers.

    T(m) = m! * 2^(m-1)
    
    :param m: Number of elements
    :return: Total number of possibilities
    """
    if m < 1:
        return 0  # No valid arrangements for m < 1
    return math.factorial(m) * (2 ** (m - 1))

# Example usage
for m in range(1, 10):
    print(f"T({m}) = {calculate_T(m)}")