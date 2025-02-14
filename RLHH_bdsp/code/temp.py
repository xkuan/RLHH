matrix = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
# matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
out = []
m = len(matrix)
n = len(matrix[0])
l, r, t, b = 0, n - 1, 0, m - 1
while len(out) < m * n:
    # left -> right (左闭右开)
    for j in range(l, r):
        out.append(matrix[t][j])
    # top to bottom
    for i in range(t, b):
        out.append(matrix[i][r])
    # right -> left
    for j in range(r, l, -1):
        out.append(matrix[b][j])
    # bottom to top
    for i in range(b, t, -1):
        out.append(matrix[i][l])

    l, r, t, b = l+1, r-1, t+1, b-1
    if l == r and t == b:
        out.append(matrix[l][t])

    print(out)
