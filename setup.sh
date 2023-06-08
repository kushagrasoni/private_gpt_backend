sudo apt update -y
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11

python3.11 --version
which python3

pip3.11 install --upgrade pip
pip3 --version
sudo apt install python3.11-pip
sudo apt install python3-pip
pip3 --version
python3.11 -m pip install --upgrade pip
python3.11 -m venv venv311_pvt_gpt
ls -ltr
rm -rf venv311_pvt_gpt/
pwd
cd ~
ls -ltr
sudo apt install python3.11-venv
python3.11 -m venv venv311_pvt_gpt
source venv311_pvt_gpt/bin/activate
which pip
pip install fastapi uvicorn
pip install -U bitsandbytes -U git+https://github.com/huggingface/transformers.git
pip install -U git+https://github.com/huggingface/peft.git -U git+https://github.com/huggingface/accelerate.git langchain
pip install -U sentence_transformers -U pypdf -U qdrant-client
ls -ltr
git clone https://github.com/kushagrasoni/private_gpt_backend.git
