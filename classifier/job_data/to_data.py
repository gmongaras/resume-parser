# Note: Data from https://www.kaggle.com/datasets/estasney/job-title-synonyms?resource=download
import re

with open("in.txt", "r")as f:
    with open("out.txt", "w") as f2:
        data = set()
        for line in f:
            for job in line.split(" => ")[0].split(", "):
                # Clean the word
                job = re.sub("[^-9A-Za-z ]", "", job.lower())
                data.add(job)

        for d in data:
            f2.write(d + "\n")
