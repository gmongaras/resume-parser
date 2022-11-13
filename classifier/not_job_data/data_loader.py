# Data from https://drive.google.com/file/d/0Bz8a_Dbh9QhbZVhsUnRWRDhETzA/view?usp=sharing&resourcekey=0-Rp0ynafmZGZ5MflGmvwLGg

import re
import random

max_num = 22000

with open('train.csv', 'r') as f:
    with open("output.txt", "w") as f2:
        output = set()
        
        for line in f:
            # Clean the line
            line = re.sub("[^-9A-Za-z ]", "", line.lower()).split(" ")

            # Get the length of the sentence / 3 possible
            # 2 or 3 words sentences
            for i in range(0, len(line)//3):
                # Get a random 2 or 3 length word sequence
                rand_len = random.randint(1, 4)
                rand_start = random.randint(0, len(line)-rand_len)
                
                # Get the random sequence
                seq = line[rand_start:rand_len+rand_start]
                
                # Save the sequence
                if len(" ".join(seq)) > 0:
                    output.add(" ".join(seq))
            if len(output) > max_num:
                break
        
        
        for s in output:
            f2.write(s + "\n")