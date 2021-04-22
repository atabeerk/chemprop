import os
from mysql.connector import connect, Error
import numpy as np
import pandas as pd
import pickle
import csv
from clustering import read_csvs



def connect_db():
    try:
        conn = connect(
            host="localhost",
            user="root",
            password="12345678",
            database="zinc15_compound_clustering"
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
    for i in range(0, 58000000, 1000000):
        query = f"SELECT * FROM compounds LIMIT {i}, {i+1000000}"
        with conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            print(result[0])
            fp = open("/VOLUMES/ATABERK64/db_chunk" + str(int(i/1000000)) + ".csv", "w+")
            f = csv.writer(fp)
            f.writerows(result)
            fp.close()
            print(str(int(i/1000000)), "complete")



    """
value_list = []
for index, row in df.iterrows():
    if index <= 84999989:
        continue
    cluster = row["labels"]
    smiles = row["smiles"]
    value_list.append([cluster, smiles])
    if len(value_list) < 100000:
        continue
    else:
        print(index)
        query = 'UPDATE compounds SET assigned_cluster=%s WHERE smiles=%s'
        with conn.cursor() as cursor:
            cursor.executemany(query, value_list)
        conn.commit()
        value_list = []
    """

    """
Left at 8599989
    """