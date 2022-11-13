"""
!pip install --upgrade pip
!pip install pyresparser
!pip install nltk
!pip install spacy==2.3.5
!pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.5/en_core_web_sm-2.3.5.tar.gz
!python -m nltk.downloader words
!python -m nltk.downloader stopwords
!pip install python-doctr
!pip install "python-doctr[torch]"
!pip install pdf2image
conda install -c conda-forge poppler
!pip install flask
"""


"""
pip install --upgrade pip
pip install pyresparser
pip install nltk
pip install spacy==2.3.5
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.5/en_core_web_sm-2.3.5.tar.gz
python -m nltk.downloader words
python -m nltk.downloader stopwords
pip install python-doctr
pip install "python-doctr[torch]"
pip install pdf2image
conda install -c conda-forge poppler
pip install flask
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
import urllib.request




def download_file(url, filename):
    # Download the file locally
    urllib.request.urlretrieve(url, filename)




app = Flask(__name__)
@app.route('/', methods = ['POST'])
def main(pdf_dict=None):
    # Get the pdf dictionary
    if pdf_dict == None:
        pdf_dict = request.args.get("url")
    else:
        url = pdf_dict["url"]
    
    # Create tmp directory
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
        
    # Download the file
    filename = "tmp/tmpfile.pdf"
    download_file(url, filename)
    
    
    
    
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
    
    # Dictionaries to store possible data
    githubs = dict()
    linkedins = dict()
    
    # What line are we currently on?
    l = 0

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
                
                # Finding the location of the person
                if location == None:
                    pos = re.findall("[\w ]+,[ ]*[A-Z]{2,}|[A-Z]{3,}", sent)
                    if len(pos) > 0:
                        location = pos[0].strip()
                
                # Sentence without spacing
                sent_no_space = sent.replace(" ", "").lower()
                
                # Find any phone numbers
                for number in re.findall("[(]?[\d]{3}[)]?[ ]?[-]?[ ]?[\d]{3}[ ]?[-]?[ ]?[\d]{4}", sent):
                    phone_numbers.append([number, l])
                
                # Find any emails
                for email in re.findall("[\w]+@[\w]+.[\w]{2,}", sent):
                    emails.append([email, l])
                
                # Find any github links in the line
                for i in range(0, len(sent_no_space)-github_window_len+1):
                    # Sentence segment
                    segment = sent_no_space[i:github_window_len+i]
                    similarity = 1
                    for j in range(0, len(segment)):
                        similarity += abs(ord(segment[j]) - ord(github_window[j]))
                    
                    # If the similarity is greater than 99%, store the sequence
                    # and it's similarity
                    if 1-(similarity/github_window_max) > 0.97:
                        # Get the github username after github.com
                        username = re.findall("[(/*)]*[/]?[\w]+", sent_no_space[github_window_len+i:])
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
                    
                    # If the similarity is greater than 99%, store the sequence
                    # and it's similarity
                    if 1-(similarity/linkedin_window_max) > 0.97:
                        # Get the linkedin username after linkedin.com/in/
                        username = re.findall("[\w]*[/]?[\w]*[/]?[\w.]+", sent_no_space[github_window_len+i:])
                        if len(username) == 0:
                            continue
                        username = username[0].split("/")[-1]
                        linkedins[linkedin_str + username] = 1-(similarity/linkedin_window_max)
                        
                        
    
    # Get the github link with the highest similarity
    try:
        github = max(githubs, key=githubs.get)
    except:
        github = None
    try:
        linkedin = max(linkedins, key=linkedins.get)
    except:
        linkedin = None
        
    # Get the earliest email
    email = None
    idx = 999999999
    for num,idx_ in emails:
        if idx_ < idx:
            email = num
        
    # Get the earliest phone number
    phone_number = None
    idx = 999999999
    for num,idx_ in phone_numbers:
        if idx_ < idx:
            phone_number = num
    
    
    output = {
        "name":name,
        "college_name":college_name,
        "degree":degree,
        "location":location,
        "email":email,
        "phone_number":phone_number,
        "linkedin":linkedin,
        "github":github,
        "skills":skills,
    }
    
    return output










if __name__ == "__main__":
    main({"url":"https://firebasestorage.googleapis.com/v0/b/hackutd-conneqt.appspot.com/o/resumes%2F38oD5dNWCmVADXuscACWdRoRGrH2%2F1668309159465.pdf?alt=media&token=a2b3ec3c-1631-4208-b1cb-81f2199dc620"})