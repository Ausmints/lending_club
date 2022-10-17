FROM python:3.8-buster
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt
COPY . .
ENTRYPOINT [ "python" ]
CMD ["flask_app.py" ]