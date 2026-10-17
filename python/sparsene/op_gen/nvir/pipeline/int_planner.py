from itertools import permutations, product
from typing import List, Tuple
import math


def is_valid_permutation(perm, constraints):
    """
    Check if a permutation satisfies the constraints.
    Each constraint is a pair (m, n) meaning m must not appear after n.
    """
    for m, n in constraints:
        if perm.index(m) > perm.index(n):  # If m appears after n, it's invalid
            return False
    return True


# 1. 我会对op做排列 -> m!
# 2. 根据opgraph的约束，剪枝掉不合法的排列 -> 得到合法的排列
# 3. 生成分隔符，插入到排列之间 -> 2^(m-1) (max_stage=4) (min_op_score_sum_in_stage=2)
# 4. 遍历可能的shifts(range=1~3)
def generate_all_partitions_with_constraints(
    m: int,
    constraints: List[Tuple[int, int]],
    min_nstage: int,
    max_nstage: int,
    min_num_op_per_stage: int,
    max_num_op_per_stage: int,
):
    # Generate all permutations of the elements 0, 1, ..., m-1
    elements = list(range(m))
    all_permutations = permutations(elements)

    # Filter permutations based on the constraints
    valid_permutations = filter(
        lambda perm: is_valid_permutation(perm, constraints), all_permutations
    )

    # For each valid permutation, generate all possible ways to insert delimiters
    results = []
    for perm in valid_permutations:
        # There are m-1 gaps between m elements, and each gap can either have a delimiter or not
        num_gaps = m - 1
        for delimiter_plan in product([0, 1], repeat=num_gaps):
            num_delimiters = sum(delimiter_plan)
            if not min_nstage <= num_delimiters + 1 <= max_nstage:
                continue
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

            num_op_per_stage_oor = False
            for stage in partition:
                if not min_num_op_per_stage <= len(stage) <= max_num_op_per_stage:
                    num_op_per_stage_oor = True
                    break
            if num_op_per_stage_oor:
                continue
            results.append(partition)

    return results


def dedup(partitions):
    deduped_partitions = []
    seen = set()  # To track unique elements

    for partition in partitions:
        # Normalize each element by converting innermost lists to sets
        normalized_partition = frozenset(frozenset(sublist) for sublist in partition)
        # Add to result if not already seen
        if normalized_partition not in seen:
            seen.add(normalized_partition)
            deduped_partitions.append(partition)

    return deduped_partitions

def calculate_possible_partitions(m):
    if m < 1:
        return 0  # No valid arrangements for m < 1
    return math.factorial(m) * (2 ** (m - 1))

if __name__ == "__main__":
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
    print("For m = 9 with constraints (nstages in [2, 3], op_per_stage in [2, 3]):")
    partitions = generate_all_partitions_with_constraints(m, constraints, 2, 3, 2, 3)
    print(f"Total number of partitions: {len(partitions)}")
    print(f"Total number of deduplicated partitions: {len(dedup(partitions))}")

    print("For m = 8 with constraints (nstages in [1, 4], op_per_stage in [1, 4]):")
    partitions = generate_all_partitions_with_constraints(m, constraints, 1, 4, 1, 4)
    print(f"Total number of partitions: {len(partitions)}")
    print(f"Total number of deduplicated partitions: {len(dedup(partitions))}")


    m = 8  # Number of elements
    constraints = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (5, 6), (6, 7)]
    print("For m = 8 with constraints (nstages in [2, 3], op_per_stage in [2, 3]):")
    partitions = generate_all_partitions_with_constraints(m, constraints, 2, 3, 2, 3)
    print(f"Total number of partitions: {len(partitions)}")
    print(f"Total number of deduplicated partitions: {len(dedup(partitions))}")

    print("Number of possible partitions:")
    for m in range(1, 10):
        print(f"T({m}) = {calculate_possible_partitions(m)}")
