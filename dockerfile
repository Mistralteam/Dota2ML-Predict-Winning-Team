FROM python:3.9.6

WORKDIR /dota2webapp

COPY dota2webapp/requirements.txt .
RUN pip install -r requirements.txt
RUN pip install pymongo
COPY dota2webapp .

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# RUN python 3typesoflearning.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
# CMD ["sh", "-c", "python 3typesoflearning.py ; flask run --host=0.0.0.0 --port=80"]
# CMD ["sh", "-c", "python deletemodelsonstartup.py ; python 3typesoflearning.py ; flask run --host=0.0.0.0 --port=80"]

