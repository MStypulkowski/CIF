FROM pytorch/pytorch:latest
USER root

RUN apt update
RUN apt install -y openssh-server sudo vim nvidia-cuda-toolkit

### Install SSH
RUN mkdir /var/run/sshd
### SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

### Change default port for SSH
RUN sed -i 's/#Port 22/Port 4444/' /etc/ssh/sshd_config
EXPOSE 4444

ARG username
ARG gid
ARG uid

RUN addgroup $username --gid $gid
RUN adduser --quiet --disabled-password --uid $uid --gid $gid $username --gecos "User"
RUN adduser $username sudo

RUN echo $username:$username | chpasswd

RUN echo export PATH=/opt/conda/bin:'$PATH' > /etc/profile.d/conda.sh
COPY environment.yaml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
RUN conda init bash
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" >> /home/$username/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

ENV PYTHONPATH /src

CMD ["/usr/sbin/sshd", "-D"]
