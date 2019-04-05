from libc.stdlib cimport malloc, free

def calculate_edit_distance(word1,  word2):
    len1 = len(word1)
    len2 = len(word2)

    cdef int** dp = <int**> malloc((len1 + 1) * sizeof(int*))
    for i in range(len1 + 1):
        dp[i] = <int*> malloc((len2 + 1) * sizeof(int))

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    cdef result = dp[len1][len2]
    for i in range(len1 + 1):
        free(dp[i])
    free(dp)

    return result