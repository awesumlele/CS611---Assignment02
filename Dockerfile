FROM apache/airflow:2.9.3-python3.12

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk-headless procps libgomp1 && \
    rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$PATH:$JAVA_HOME/bin

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY --chown=airflow:root dags    /opt/airflow/dags
COPY --chown=airflow:root scripts /opt/airflow/scripts
COPY --chown=airflow:root utils   /opt/airflow/utils
COPY --chown=airflow:root data    /opt/airflow/data

ENV PIPELINE_ROOT=/opt/airflow \
    PIPELINE_SCRIPTS=/opt/airflow/scripts \
    PYTHONPATH=/opt/airflow:/opt/airflow/scripts \
    AIRFLOW__CORE__LOAD_EXAMPLES=False
