"""
!pip install --upgrade pip
!pip install numpy
!pip install torch
!pip install pyresparser
!pip install nltk
!pip install thinc==7.4.1
!pip install spacy==2.3.5
!pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
!python -m nltk.downloader words
!python -m nltk.downloader stopwords
!python -m nltk.downloader punkt
!pip install python-doctr
!pip install "python-doctr[torch]"
!pip install pdf2image
!conda install -c conda-forge poppler
!pip install flask
!pip install pyOpenSSL
!pip install flask-cors
"""


"""
pip install --upgrade pip
pip install numpy
pip install torch
pip install pyresparser
pip install nltk
pip install thinc==7.4.1
pip install spacy==2.3.5
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
python -m nltk.downloader words
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
pip install python-doctr
pip install "python-doctr[torch]"
pip install pdf2image
conda install -c conda-forge poppler
pip install flask
pip install pyOpenSSL
pip install flask-cors
"""



import spacy
from pyresparser import ResumeParser

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from pdf2image import convert_from_path
import os
import re

from flask import Flask
from flask import request
from flask_cors import CORS
import urllib.request

import nltk
from nltk.tag import StanfordNERTagger

import torch
import json
import numpy as np


    
    
    
    
    
    
    
    
###### Job Classifier

# Load vocabs
with open("classifier/models/vocab.json", "r") as file :
    vocab = json.load(file)
with open("classifier/models/vocab_inv.json", "r") as file:
    vocab_inv = json.load(file)

# Classifier model
class model(torch.nn.Module):
    def __init__(self, in_, out):
        super(model, self).__init__()
        
        self.zero = torch.nn.Embedding(in_, in_)
        self.first = torch.nn.LSTM(in_, out, num_layers=2, bidirectional=True)
        self.second = torch.nn.Sequential(
            torch.nn.Linear(out*2, 1),
            torch.nn.Sigmoid()
        )
        
        self.bce = torch.nn.BCELoss(reduction="none")
        
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0005)
    
    
    def forward(self, X):
        return self.second(self.first(self.zero(X))[0])[:, 0].squeeze(-1)
    
# Load model state
M = model(len(vocab), 16)
M.load_state_dict(torch.load("classifier/models/model1.pkl"))
M.eval()
max_len = 88
PAD_tok = torch.tensor(vocab["<PAD>"]).int()



# Person tagger
tagger = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')







app = Flask(__name__)
CORS(app)
@app.route('/', methods = ['POST'])
def main(pdf_dict=None, isURL=True, java_path=None):
    if java_path != None:
        os.environ['JAVAHOME'] = java_path
        nltk.internals.config_java(java_path)
    
    # Get the pdf dictionary
    if pdf_dict == None:
        pdf_dict = request.get_json()
        
        # Get the url
        url = pdf_dict["url"]
    elif type(pdf_dict) == dict:
        # Get the url
        url = pdf_dict["url"]
    elif type(pdf_dict) == str:
        filename = url = pdf_dict
    else:
        print("Input type not supported")
    
    # Create tmp directory
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
        
    # Download the file
    if isURL:
        filename = "tmp/tmpfile.pdf"

        # Download the file locally
        urllib.request.urlretrieve(url, filename)
    else:
        filename = url
    
    
    
    
    #### Model 1
    # Repo: https://github.com/OmkarPathak/pyresparser
    
    # Use the first model to get some data
    data = ResumeParser(filename).get_extracted_data()

    # Get the name and email
    name = data["name"]
    college_name = data["college_name"]
    degree = data["degree"]
    emails = []
    phone_numbers = []
    twitter = []
    skills = data["skills"]
    
    
    
    
    
    #### Model 2
    """
    Repo: https://github.com/mindee/doctr
    Documentation: https://mindee.github.io/doctr/modules/io.html#document-structure
    
    det_arch models:
    https://mindee.github.io/doctr/using_doctr/using_models.html#available-architectures
      linknet_resnet18
      db_resnet50               (default)
      db_mobilenet_v3_large
      linknet_resnet18_rotation
      db_resnet50_rotation
      
    reco_arch models:
    https://mindee.github.io/doctr/using_doctr/using_models.html#text-recognition
      crnn_vgg16_bn             (default)
      crnn_mobilenet_v3_small
      crnn_mobilenet_v3_large
      sar_resnet31
      master
    """
    
    # Using the normal document parser, get all data from the document
    model = ocr_predictor(det_arch="db_mobilenet_v3_large", reco_arch="crnn_vgg16_bn", pretrained=True, export_as_straight_boxes=True)
    
    # Create images from the pdf
    images = convert_from_path(filename, dpi=400)
    
    # Save the images to a tmp directory
    paths = [f"tmp/tmp{i}.jpg" for i in range(0, len(images))]
    for img, path in zip(images, paths):
        img.save(path)
        
    # Load in the images
    pdf_doc = DocumentFile.from_images(paths)
    data = model(pdf_doc).pages
    
    # Delete the files in the temporary directory
    for path in paths:
        os.remove(path)
    
    # string windows for linkedin and github
    linkedin_window = "inkedin.com"
    linkedin_str = "linkedin.com/in/"
    linkedin_window_len = len(linkedin_window)
    linkedin_window_max = len(linkedin_window)*ord("a")
    github_window = "github.com"
    github_window_len = len(github_window)
    github_window_max = len(github_window)*ord("a")
    twitter_window = "twitter.com"
    twitter_window_len = len(twitter_window)
    twitter_window_max = len(twitter_window)*ord("a")
    
    # Dictionaries to store possible data
    githubs = dict()
    linkedins = dict()
    twitters = dict()
    job_titles = dict()
    
    # Name of the person as a list
    person_list = list()
    person_list_type = list()
    
    # What line are we currently on?
    l = 0
    
    # What word are we currently on?
    w = 0

    # Location of the person
    location = None

    # Iterate over all pages
    for page in data:
        # Iterate over all blocks in the page
        for block in page.blocks:
            # Iterate over all lines in the block
            for line in block.lines:
                l+=1
                
                # Combine the words into a sentence
                sent = [sent.value for sent in line.words]
                sent = " ".join(sent)
                
                # Find the name
                if len(person_list) < 3 and w < 10:
                    for sent in nltk.sent_tokenize(sent):
                        if len(person_list) == 2 and person_list_type[0] == True and person_list_type[1] == True:
                            break
                        tokens = nltk.tokenize.word_tokenize(sent)
                        try:
                            tags = tagger.tag(tokens)
                        except LookupError:
                            java_path = "C:\Program Files\Java\jdk-18\bin\java.exe"
                            os.environ['JAVAHOME'] = java_path

                            tags = tagger.tag(tokens)
                        for tag in tags:
                            if (len(person_list) >= 3) or \
                                (len(person_list) == 2 and person_list_type[0] == True and person_list_type[1] == True):
                                break
                            
                            if tag[1]=='PERSON' and len(tag[0]) > 1:
                                person_list.append(tag[0])
                                person_list_type.append(True)
                            elif len(person_list) > 0 and person_list_type[0] == True:
                                person_list.append(tag[0])
                                person_list_type.append(False)
                            else:
                                person_list = []
                                person_list_type = []
                            
                            w += 1
                
                # Finding the location of the person
                if location == None:
                    pos = re.findall("[\w ]+[,]{1}[ ]*[[A-Z]{2,}|[A-Z]{3,}]", sent)
                    if len(pos) > 0:
                        location = pos[0].strip()
                
                # Sentence without spacing
                sent_no_space = sent.replace(" ", "").lower()
                
                # Sentence as a list
                sent_list = sent.split(" ")
                
                # Find any phone numbers
                for number in re.findall("[(]?[\d]{3}[)]?[ ]?[-]?[ ]?[\d]{3}[ ]?[-]?[ ]?[\d]{4}", sent):
                    phone_numbers.append([number, l])
                
                # Find any emails
                for email in re.findall("[\w.]+@[\w]+.[\w.]{2,}", sent):
                    emails.append([email, l])
                
                # Find any github links in the line
                for i in range(0, len(sent_no_space)-github_window_len+1):
                    # Sentence segment
                    segment = sent_no_space[i:github_window_len+i]
                    similarity = 1
                    for j in range(0, len(segment)):
                        similarity += abs(ord(segment[j]) - ord(github_window[j]))
                    
                    # If the similarity is greater than 97%, store the sequence
                    # and it's similarity
                    if 1-(similarity/github_window_max) > 0.97:
                        # Get the github username after github.com
                        username = re.findall("[(/*)]*[/]?[\w-]+", sent_no_space[github_window_len+i:])
                        if len(username) == 0:
                            continue
                        username = username[0]
                        if username[0] != "/":
                            username = "/" + username
                        githubs[github_window + username] = 1-(similarity/github_window_max)
                
                # Find any linkedin links in the line
                for i in range(0, len(sent_no_space)-linkedin_window_len+1):
                    # Sentence segment
                    segment = sent_no_space[i:linkedin_window_len+i]
                    similarity = 1
                    for j in range(0, len(segment)):
                        similarity += abs(ord(segment[j]) - ord(linkedin_window[j]))
                    
                    # If the similarity is greater than 97%, store the sequence
                    # and it's similarity
                    if 1-(similarity/linkedin_window_max) > 0.97:
                        # Get the linkedin username after linkedin.com/in/
                        username = re.findall("[\w]*[/]?[\w]*[/]?[\w.-]+", sent_no_space[linkedin_window_len+i:])
                        if len(username) == 0:
                            continue
                        username = username[0].split("/")[-1]
                        linkedins[linkedin_str + username] = 1-(similarity/linkedin_window_max)
                
                
                # Find any twitter links in the line
                for i in range(0, len(sent_no_space)-twitter_window_len+1):
                    # Sentence segment
                    segment = sent_no_space[i:twitter_window_len+i]
                    similarity = 1
                    for j in range(0, len(segment)):
                        similarity += abs(ord(segment[j]) - ord(twitter_window[j]))
                    
                    # If the similarity is greater than 97%, store the sequence
                    # and it's similarity
                    if 1-(similarity/twitter_window_max) > 0.97:
                        # Get the twitter username after twitter.com/
                        username = re.findall("[/]*[@]?[\w]*", sent_no_space[linkedin_window_len+i:])
                        if len(username) == 0:
                            continue
                        username = username[0].replace(" ", "")
                        if len(username) == 0:
                            continue
                        if username[0] != "/":
                            username = "/" + username
                        twitters[twitter_window + username] = 1-(similarity/twitter_window_max)
                
                

                # Find the job title in the first L lines
                L = 100
                if l < L:
                    # Segments in the line with 2 words
                    segments_enc = [
                        torch.tensor(
                            [vocab[j] for j in
                                re.sub("[^-9A-Za-z ]", "", " ".join(sent_list[i:i+2]).lower())
                            ]
                        ) for i in range(0, len(sent_list)-1)
                    ]
                    segments = [
                        re.sub("[^-9A-Za-z ]", "", " ".join(sent_list[i:i+2])).lower()
                        for i in range(0, len(sent_list)-1)
                    ]
                    
                    # Segments in the line with 1 word
                    segments_enc += [
                        torch.tensor(
                            [vocab[j] for j in
                                re.sub("[^-9A-Za-z ]", "", " ".join(sent_list[i:i+1]).lower())
                            ]
                        ) for i in range(0, len(sent_list)-1)
                    ]
                    segments += [
                        re.sub("[^-9A-Za-z ]", "", " ".join(sent_list[i:i+1])).lower()
                        for i in range(0, len(sent_list)-1)
                    ]
                    
                    # Padding
                    for i in range(0, len(segments_enc)):
                        segments_enc[i] = torch.cat((segments_enc[i], PAD_tok.repeat(max_len-len(segments_enc[i]))), dim=0)

                    try:
                        # Convert the segments to a batch
                        segments_enc = torch.stack(segments_enc).int()

                        # Get the model output on the batch
                        segments_out = M(segments_enc)

                        # Save the predictions
                        for i in range(0, len(segments)):
                            job_titles[segments[i]] = segments_out[i].item()
                    except RuntimeError:
                        pass


    # Delete the files in the temporary directory
    if isURL:
        os.remove(filename)
                  
                    
                    
                    
    # Get the links with the highest similarity
    try:
        github = max(githubs, key=githubs.get)
    except:
        github = None
    try:
        linkedin = max(linkedins, key=linkedins.get)
    except:
        linkedin = None
    try:
        twitter = max(twitters, key=twitters.get)
    except:
        twitter = None
        
    # Get the earliest email
    email = None
    idx = np.inf
    for num,idx_ in emails:
        if idx_ < idx:
            email = num
        
    # Get the earliest phone number
    phone_number = None
    idx = np.inf
    for num,idx_ in phone_numbers:
        if idx_ < idx:
            phone_number = num
            
    # Get the highest confidence prediction for the job
    job_title_max = np.argmax(list(job_titles.values()))
    job_title = list(job_titles.keys())[job_title_max]
    
    about_me = None
    
    # If a person was found, save it
    if len(person_list) == 3:
        name = person_list[0] + " " + person_list[2]
    elif len(person_list) == 2:
        name = " ".join(person_list)
    
    output = {
        "name":name if name != None else "",
        "college_name":college_name if college_name != None else "",
        "degree":degree if degree != None else "",
        "location":location if location != None else "",
        "email":email if email != None else "",
        "phone_number":phone_number if phone_number != None else "",
        "linkedin":linkedin if linkedin != None else "",
        "github":github if github != None else "",
        "twitter":twitter if twitter != None else "",
        "skills":skills if skills != None else "",
        "about_me":about_me if about_me != None else "",
        "job_title":job_title if job_title != None else "",
    }
    
    return output









if __name__ == "__main__":
    # print(main({"url":"https://firebasestorage.googleapis.com/v0/b/hackutd-conneqt.appspot.com/o/resumes%2F38oD5dNWCmVADXuscACWdRoRGrH2%2F1668309159465.pdf?alt=media&token=a2b3ec3c-1631-4208-b1cb-81f2199dc620"}))
    # print(main("./test.pdf", False, java_path = "C:/Program Files/Java/jdk-18/bin/java.exe"))
    print(main("https://www.pdf-archive.com/2017/09/26/fake-resume/fake-resume.pdf", True, java_path = "C:/Program Files/Java/jdk-18/bin/java.exe"))
          
