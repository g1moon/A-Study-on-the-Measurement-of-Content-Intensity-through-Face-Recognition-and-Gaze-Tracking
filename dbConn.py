import sqlite3 as db
import numpy as np
import pandas as pd

# conn 함수 : Conn
# disconn 함수 : DisConn
# insert 함수 : InsertLog
# select 함수 : SelectLog

# 광고 시청자 정보 : avgPeople
# 광고 시청자 정보

global conn
global c

#path = './DaFarm/testDB.sqlite'

# conn 함수
def Conn(path) :
    global conn
    global c
    conn = db.connect(path)
    c = conn.cursor()

# disconn 함수
def DisConn() :
    global conn
    global c
    c.close()
    conn.close()

# insert 함수
'''Sex 성별 / Age 나이 / Watch 응시시간 / Time 시간 : PK / Station 역이름 / AD_ID 광고ID'''
def InsertLog(arg_list) :
    global conn
    global c
    
    age = arg_list[1]
    if 0 <= age < 10:
        age = '09'
    elif 10 <= age < 20:
        age = '1019'
    elif 20 <= age < 40:
        age = '2039'
    elif 40 <= age < 60:
        age = '4059'
    elif 60 <= age:
        age = '60'
    else:
        age = '100'

    sql_command = 'INSERT INTO AD_LOG VALUES(' + '"' + str(arg_list[0]) + '"' \
                                               ', "' + str(arg_list[1]) + '"' \
                                               ', "' + str(arg_list[2]) + '"' \
                                               ',STRFTIME("%Y%m%d%H%M%f", "NOW", "LOCALTIME")' \
                                               ', "' + str(arg_list[3]) + '"' \
                                               ', "' + str(arg_list[4]) + '");'

    c.execute(sql_command)
    conn.commit()
    print("AD_ID : {}, Station : {}, Sex : {}, Age : {} insert commit !!".format(arg_list[4], arg_list[3], arg_list[0], arg_list[1]))

# select 함수
def SelectLog(subNm, orderBy) :
    global conn
    global c
    order_sql = ''
    search_key = ''
    cnt = 0

    for i in subNm :
        cnt += 1
        if(cnt == 1) :
            search_key += '"' + i + '"'
        else :
            search_key += ', "' + i + '"'

    sql_command = 'SELECT Time'     \
                  '      ,Sex'      \
                  '      ,Age'      \
                  '      ,Watch'    \
                  '      ,Station'  \
                  '      ,AD_ID '   \
                  ' FROM AD_LOG '   \
                  'WHERE Station IN ('+ search_key +')'
    if orderBy == 0 : # 오름차순
        order_sql = ' ORDER BY TIME ASC'
    else : # 내림차순
        order_sql = ' ORDER BY TIME DESC'

    sql_command += order_sql

    print(sql_command)
    c.execute(sql_command)
    res = c.fetchall()
    if(res != None) :
        for i in res :
            print(i)


def avgPeople(df):
    df_pi = pd.crosstab(df['frameId'], df['gender'])
    df_pi['gender'] = df_pi['F'] < df_pi['M']  # True = M, False = F
    df_pi.drop(['F', 'M'], axis=1, inplace=True)

    df_gp = df.groupby(['frameId']).mean()
    df_gp.drop(['eyetime'], axis=1, inplace=True)

    df_gaze = df.groupby(['frameId', 'subNM']).sum()
    df_gaze.reset_index(inplace=True)
    df_gaze.drop(['age', 'AD_ID', 'frameId'], axis=1, inplace=True)

    result = pd.concat([df_pi, df_gp, df_gaze], axis=1, join_axes=[df_pi.index])
    result.reset_index(inplace=True)

    data_cnt = len(result)
    arg_list = list()

    if data_cnt > 0:
        for id_cnt in range(0, data_cnt):

            gender = ''
            age = ''

            if result.loc[id_cnt, 'gender']:
                gender = 'M'
            else:
                gender = 'F'

            if 0 <= result.loc[id_cnt, 'age'] < 10:
                age = '09'
            elif 10 <= result.loc[id_cnt, 'age'] < 20:
                age = '1019'
            elif 20 <= result.loc[id_cnt, 'age'] < 40:
                age = '2039'
            elif 40 <= result.loc[id_cnt, 'age'] < 60:
                age = '4059'
            elif 60 <= result.loc[id_cnt, 'age']:
                age = '60'
            else:
                age = '100'

            arg_list.append([gender
                                , str(age)
                                , str(result.loc[id_cnt, 'eyetime'])
                                , result.loc[id_cnt, 'subNM']
                                , str(result.loc[id_cnt, 'AD_ID'])])

    Conn()
    for arg in arg_list:
        # print(arg)
        InsertLog(arg)
    DisConn()

    return 0
