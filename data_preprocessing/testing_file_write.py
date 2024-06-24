text_file = r'C:\Users\2418015\Downloads\final_data\128_testing\test.txt'

with open(text_file, 'w') as f:
    for a in range(42):
        for j in range(10):
            f.write("'{}_{}'\n".format(a, j))
    for b in range(330):
        w = b+a
        for j in range(10):
            f.write("'{}_{}'\n".format(w, j))
    for c in range(92):
        w = a+b+c
        for j in range(15):
            f.write("'{}_{}'\n".format(w, j))