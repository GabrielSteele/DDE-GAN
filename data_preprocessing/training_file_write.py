text_file = r'C:\Users\2418015\Downloads\final_data\128_training\train.txt'

with open(text_file, 'w') as f:
    for a in range(45):
        for j in range(10):
            f.write("'{}_{}'\n".format(a, j))
    for b in range(48):
        w = b+a
        for j in range(10):
            f.write("'{}_{}'\n".format(w, j))
    for c in range(32):
        w = a+b+c
        for j in range(15):
            f.write("'{}_{}'\n".format(w, j))
    for d in range(41):
        w = a+b+c+d
        for j in range(15):
            f.write("'{}_{}'\n".format(w, j))
    for e in range(13):
        w = a+b+c+d+e
        for j in range(10):
            f.write("'{}_{}'\n".format(w, j))
    for g in range(160):
        w = a+b+c+d+e+g
        for j in range(15):
            f.write("'{}_{}'\n".format(w, j))