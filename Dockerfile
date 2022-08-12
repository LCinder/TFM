FROM python:3.9-alpine
WORKDIR /src
COPY requirements.txt /src/
RUN pip install -r requirements.txt
COPY . /src
CMD ["python", "wsgi.py"]