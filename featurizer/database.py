from mysql.connector import connect, Error
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
    insert_compounds = "INSERT IGNORE INTO compounds (smiles, featurized)" \
                       "VALUES (%s, %s)"

    df = read_csvs(folder="multi_task_features_dmpnn_25_zinc_flagments")
    """
value_list = []
i = 1
for index, row in df.iterrows():
    value_list.append((row["smiles"], row["descriptors"].dumps()))
    if len(value_list) == 100000:
        with conn.cursor() as cursor:
            cursor.executemany(insert_compounds, value_list)
            conn.commit()
        value_list = []
        print(i*100000)
        i += 1
    """