FROM python:3.12

WORKDIR /code

COPY . /code/fastapi

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["python", "fastapi/main.py"]