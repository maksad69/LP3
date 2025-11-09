def knapsack_01(values, weights, capacity):
    n = len(values)
    
    capacity = int(capacity)
    
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    print("\nDP Matrix:")
    for row in dp:
        print(row)

    selected_items = []
    i, w = n, capacity
    while i > 0 and w > 0:
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]
        i -= 1

    selected_items.reverse()

    return dp[n][capacity], selected_items

if __name__ == "__main__":
    n = int(input("Enter the number of items: "))  # e.g., 3
    values = []
    weights = []
    
    for i in range(n):
        value = int(input(f"Enter the value of item {i + 1}: "))  # e.g., 60, 100, 120
        weight = int(input(f"Enter the weight of item {i + 1}: "))  # e.g., 10, 20, 30
        values.append(value)
        weights.append(weight)

    max_capacity = int(input("Enter the maximum capacity of the knapsack: "))  # e.g., 50

    max_value, selected_items = knapsack_01(values, weights, max_capacity)

    print("\nSelected Items:")
    for item in selected_items:
        print(f"Item {item + 1} with weight {weights[item]} and value {values[item]}")

    print(f"\nMaximum value achievable: {max_value}")
