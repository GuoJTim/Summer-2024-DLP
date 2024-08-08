# script for drawing figures, and more if needed

from Dataloader import *
import matplotlib.pyplot as plt
import itertools

database = None
def connect_db():
    global database
    import pymysql
    database = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='',
        db='lab02',
        charset='utf8'
    )


def insert_data(model_name, epoch, loss, accuracy, batch_size, Nu, Nt, learning_rate, optimizer, train_method, note="",reset=False):
    global database
    import pymysql
    if (database == None):
        connect_db()


    cursor = database.cursor()
    sql = """
    INSERT INTO sccnet (model_name, epoch, loss, accuracy, batch_size, Nu, Nt, learning_rate, optimizer, train_method, note)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (model_name, epoch, loss, accuracy, batch_size, Nu, Nt, learning_rate, optimizer, train_method, note))
    database.commit()

def get_last_epoch(model_name,batch_size, Nu, Nt, learning_rate, optimizer, train_method):
    global database
    import pymysql
    cursor = database.cursor(pymysql.cursors.DictCursor)
    sql = "SELECT MAX(epoch) as last_epoch FROM sccnet WHERE model_name = %s AND batch_size = %s AND Nu = %s AND Nt = %s AND learning_rate = %s AND optimizer = %s AND train_method = %s"
    cursor.execute(sql, (model_name,batch_size, Nu, Nt, learning_rate, optimizer, train_method))
    result = cursor.fetchone()
    return int(result['last_epoch']) if result['last_epoch'] is not None else 0


# train_method
# SD_train
# LOSO
# LOSO_FT
def fetch_data(train_method,addition_cond):
    global database
    import pymysql
    if (database == None):
        connect_db()
    cursor = database.cursor(pymysql.cursors.DictCursor)
    sql = "SELECT model_name, accuracy, epoch, loss, Nu, Nt, learning_rate, optimizer, batch_size FROM sccnet WHERE train_method = '"+str(train_method)+"' "+addition_cond+" AND epoch % 20 = 0 AND epoch <= 300 GROUP BY model_name,epoch,Nu, Nt, learning_rate, optimizer, batch_size"
    cursor.execute(sql)
    return cursor.fetchall()


def plot_data(train_method,addition_cond=""):
    data = fetch_data(train_method,addition_cond)
    plt.figure(figsize=(12, 8))
    grouped_data = {}
    for row in data:
        key = (row['Nu'], row['Nt'], row['learning_rate'], row['optimizer'], row['model_name'], row['batch_size'])
        if key not in grouped_data:
            grouped_data[key] = {'epochs': [], 'accuracy': []}
        grouped_data[key]['epochs'].append(row['epoch'])
        grouped_data[key]['accuracy'].append(row['accuracy'])

    for key, values in grouped_data.items():
        label = f"Model={key[4]}, Nu={key[0]}, Nt={key[1]}, LR={key[2]}, Opt={key[3]}, BZ={key[5]}"
        plt.plot(values['epochs'], values['accuracy'], label=label)

    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title(train_method)
    plt.legend()
    plt.show()




def fetch_data2(train_method,addition_cond):
    global database
    import pymysql
    if (database == None):
        connect_db()
    cursor = database.cursor(pymysql.cursors.DictCursor)
    sql = "SELECT model_name, accuracy, epoch, loss, Nu, Nt, learning_rate, optimizer, batch_size FROM sccnet WHERE train_method = '"+str(train_method)+"' "+addition_cond+" GROUP BY model_name,epoch,Nu, Nt, learning_rate, optimizer, batch_size"
    cursor.execute(sql)
    return cursor.fetchall()


def plot_bar_data(train_method, x_axis='epoch', addition_cond="",width=0.1):
    # 獲取數據
    data = fetch_data2(train_method, addition_cond)
    
    # 創建圖形
    plt.figure(figsize=(16, 8))  # 增大圖形尺寸以容納更多標籤
    
    # 分組數據
    grouped_data = {}
    for row in data:
        key = (row['model_name'],row['Nu'], row['Nt'], row['learning_rate'], row['optimizer'], row['batch_size'])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(row)
    
    # 繪製長條圖
    #width = 0.05  # 條形寬度
    x_ticks = sorted(set([row[x_axis] for row in data]))
    x_positions = np.arange(len(x_ticks), dtype=np.float64)  # x 軸刻度的位置
    last_pos = np.zeros(len(x_ticks))  # 初始化位置


    # 生成顏色列表
    color_map = plt.get_cmap('tab20')  # 使用預設顏色映射
    colors = itertools.cycle(color_map.colors)  # 循環使用顏色
    
    legend_handles = {}  # 用來存儲圖例句柄
    legend_labels = []   # 用來存儲圖例標籤
    color_map = {}       # 用於記錄每個 filtered_key 對應的顏色
    
    for i, (key, values_list) in enumerate(grouped_data.items()):
        accuracies = {x_tick: [] for x_tick in x_ticks}
        
        for values in values_list:
            x_value = values[x_axis]
            accuracies[x_value].append(values['accuracy'])
        # 去除 x_axis 部分的圖例標籤
        filtered_key = ', '.join(f'{k}={v}' for k, v in zip(['model_name','Nu', 'Nt', 'learning_rate', 'optimizer', 'batch_size'], key) if k != x_axis)
        
        # 確定顏色
        if filtered_key not in color_map:
            color = next(colors)
            color_map[filtered_key] = color
        else:
            color = color_map[filtered_key]
        
        # 繪製每個 x 軸刻度的條形圖
        for j, x_tick in enumerate(x_ticks):
            max_accuracy = max(accuracies[x_tick]) if accuracies[x_tick] else 0
            if max_accuracy == 0:
                continue
            
            bar_position = x_positions[j] + last_pos[j]
            plt.bar(bar_position, max_accuracy, width, color=color)
            last_pos[j] += width
        
        # 確保圖例顯示正確
        if filtered_key not in legend_handles:
            legend_handles[filtered_key] = plt.Line2D([0], [0], color=color, lw=4)
            #if ()
            #label = f"Model={key[4]}, Nu={key[0]}, Nt={key[1]}, LR={key[2]}, Opt={key[3]}, BZ={key[5]}"
            label = filtered_key.replace("model_name","Model")
            label = label.replace("learning_rate","LR")
            label = label.replace("optimizer","Opt")
            legend_labels.append(f'{label}')
    
    # 設置圖表
    plt.xlabel(x_axis)
    plt.ylabel('Accuracy')
    plt.title(train_method)
    plt.xticks(ticks=x_positions, labels=x_ticks, rotation=45, ha='right')  # 旋轉標籤並調整對齊
    plt.legend(handles=list(legend_handles.values()), labels=legend_labels, title='Data Points', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # 自動調整佈局以適應標籤
    plt.show()



def fetch_data3(train_method,addition_cond):
    global database
    import pymysql
    if (database == None):
        connect_db()
    cursor = database.cursor(pymysql.cursors.DictCursor)
    sql = "SELECT model_name, max(accuracy) FROM sccnet WHERE train_method = '"+str(train_method)+"' "+addition_cond+" GROUP BY train_method"
    cursor.execute(sql)
    return cursor.fetchall()



def plot_single_bar_chart():
    # 示例數據
    categories = ['SD', 'LOSO', 'LOSO+FT']
    
    values = [0.8, 0.6, 0.75]  # 每個類別對應的值
    i = 0
    for label in ['SD','LOSO','FT']:
        data = fetch_data3(label,"")
        values[i] = data[0]['max(accuracy)']
        i += 1
    # 創建圖形
    print(values)
    plt.figure(figsize=(8, 6))

    # 繪製條形圖
    plt.bar(categories, values, color='green')

    # 設置標籤和標題
    plt.xlabel('Train method')
    plt.ylabel('Accuracy')
    plt.title('Comparison between the three training methods')

    # 顯示圖形
    plt.tight_layout()  # 自動調整佈局以適應標籤
    plt.show()
