import os
import pickle
import time

from mysql.connector import connect, Error
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

from csv_reader import read_csv




def connect_db():
    try:
        conn = connect(
            host="localhost",
            user="root",
            password="12345678",
            database="zinc15_compound_clustering",
            connection_timeout=600
        )
        return conn
    except Error as e:
        print(e)


def execute_query(conn, query):
    with conn.cursor() as cursor:
        cursor.execute(query)
        conn.commit()


if __name__ == "__main__":
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT features, assigned_cluster FROM compounds WHERE assigned_cluster IS NOT NULL")
    result = cursor.fetchall()

    labels = np.zeros(len(result))
    features = np.zeros([len(result), 25])
    for i in range(features.shape[0]):
        if i % 100000 == 0:
            print(i)
        features[i] = pickle.loads(result[i][0])
        labels[i] = result[i][1]



    print("starting silhouette index")
    ss = silhouette_score(features, labels, sample_size=1000000)