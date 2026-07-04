FROM python:3.12.3-bullseye

RUN pip install poetry
RUN poetry self add poetry-plugin-export
RUN poetry config virtualenvs.create false

WORKDIR /code
COPY ./pyproject.toml /code/
RUN poetry export --without-hashes --dev --output requirements.txt
RUN pip install -r requirements.txt

COPY . /code/
ENV PYTHONPATH=/code

CMD ["python", "submissions/src/main.py", "--method", "fpfh"]
