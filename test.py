from collections import Counter

def has_over_n_distinct_elements(input_list, N, m):
    """
    Checks if there are more than N distinct elements in the list, 
    where each distinct element appears more than m times.

    Args:
        input_list (list): A list of elements (elements may be lists).
        N (int): The threshold for the number of distinct elements.
        m (int): The minimum number of occurrences for an element to count as distinct.

    Returns:
        bool: True if there are more than N distinct elements that appear more than m times, False otherwise.
    """
    from collections import Counter

    # Convert all elements to tuples (if they are lists) to make them hashable
    hashable_elements = [tuple(elem) if isinstance(elem, list) else elem for elem in input_list]
    print(f"hashable_elements: {hashable_elements}")
    # Count occurrences of each element
    element_counts = Counter(hashable_elements)
    print(f"element_counts: {element_counts}")

    # Filter elements that appear more than m times
    frequent_elements = [elem for elem, count in element_counts.items() if count >= m]
    print(f"frequent_elements: {frequent_elements}")

    # Check if the number of distinct frequent elements exceeds N
    return len(frequent_elements) >= N


# Example usage:
list_of_lists1 = [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [5, 6]]
list_of_lists2 = [[1, 2], [1, 2], [1, 2], [5, 6], [1, 2]]
list_of_lists3 = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
N = 2
print(has_over_n_distinct_elements(list_of_lists3, N, 2))  # Output: True
# print(has_over_n_distinct_elements(list_of_lists2, N,2))  # Output: True
# print(has_over_n_distinct_elements(list_of_lists3, N,2))  # Output: True
