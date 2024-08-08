
import matplotlib.pyplot as plt
import itertools

database = None
def connect_db():
    global database
    import pymysql
    database = pymysql.connect(
        host='192.168.121.41',
        port=3306,
        user='lab04',
        passwd='lab04',
        db='lab04',
        charset='utf8'
    )


def insert_data(save_root, epoch, avg_loss, avg_psnr,type):
    global database
    import pymysql
    if (database == None):
        connect_db()
    #type val or train
    cursor = database.cursor()
    sql = """
    INSERT INTO vae (save_root, epoch, avg_loss, avg_psnr,type)
    VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (save_root, epoch, avg_loss, avg_psnr,type))
    database.commit()