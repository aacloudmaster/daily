################################ For Locall ################################
FROM python:3

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python", "./app_local_test.py" ]


################################ For Cloud ################################
#FROM python:3

#WORKDIR /app

#COPY requirements.txt ./

#RUN pip install --no-cache-dir -r requirements.txt

#COPY app_cloud.py app.py

#EXPOSE 5000

#CMD [ "python", "app.py" ]
