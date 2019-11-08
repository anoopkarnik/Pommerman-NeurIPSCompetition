FROM python:3.6

ADD collect_data.py /

ADD imitation.py /

ADD train.py /

ADD run_script.sh /

ADD run.py /

RUN pip install --upgrade pip

RUN git clone https://github.com/MultiAgentLearning/playground ~/playground

RUN cd ~/playground && pip install -U .

RUN pip install torch

RUN pip install gym 

RUN pip install matplotlib

CMD [ "bash", "run_script.sh"] 
