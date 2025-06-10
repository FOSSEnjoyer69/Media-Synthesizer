### External Requirments
1. python3.12 or silimar

### Installation
1. cd into the directory you want to download this programm to
2. Run the code below
```bash
git clone https://github.com/FOSSEnjoyer69/Media-Synthesizer.git
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Running
1. Make sure you have linked the interperator to the virtual enviornment
2. Run the code below
```bash
python3 src/main.py
```
#### SEVERE WARNINGS
<ol>
    <li>Stable Cascade currently causes a memory leak causing a system freeze, the memory can be cleared by closeing the program before the system freeze</li>
</ol>

### Limitations
<ol>
  <li>Limited features</li>
  <li>Only supported Image Inpainting (Stable Diffusion Inpaint) and text2image with Stable Cascade, i am working on more</li>
</ol>

### Known Issues
<ol>
  <li>Image Editors not working on Librewolf</li>
</ol>
