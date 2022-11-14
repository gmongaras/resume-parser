# Description
This project is a resume parser that I built in 24 hrs for a HackUTD 2022. It parses a pdf resume and extracts some information from it.




# Usage


## Dependencies
Before running the script, make sure to install the needed packages:
```
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
```

<b>Note</b>: A lot of packages are needed and it may break some dependencies. I ran into a few issues and found that making a new venv for this project works best.
<b?Note</b>: I use conda to install poppler which is a needed packages, but you may have to install it another way: https://poppler.freedesktop.org/



## Running the script

The script can be run using the following command:

`python app.py`

Notice how the script is named app.py, so it has Flask support. During the hackathon, I used it as a backend which is why there is Flask support on the script.

To edit the script, open app.py in a text editor and edit the `if __name__ == '__main__'` part of it. The script has three paramters:
1. pdf_dict - This can be:
   1. A URL to an online pdf: `https://www.pdf-archive.com/2017/09/26/fake-resume/fake-resume.pdf`
   2. A dictionary with the url: `{'url':https://www.pdf-archive.com/2017/09/26/fake-resume/fake-resume.pdf}`
   3. A relative or absolute path to a pdf resume on your system: `./test.pdf`
2. (Optional) isURL - (Defaults to True) True if the first paramter is a URL (options 1 or 2), False if the first paramter is a path (option 3)
3. (Optional) java_path - On my system, I receive an error stating `NLTK was unable to find the java file!` If this happens, you may have to enter the path to you java.exe file on your system like the following `C:/Program Files/Java/jdk-18/bin/java.exe`