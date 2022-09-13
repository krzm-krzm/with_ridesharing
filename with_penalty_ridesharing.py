import numpy as np
import pandas as pd
import networkx as nx
import math
from itertools import product
import matplotlib.pyplot as plt
import random
import copy
import time

def distance(x1, x2, y1, y2):
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d


def Setting(FILENAME):
    mat = []
    with open('/home/kurozumi/デスクトップ/benchmark2/' + FILENAME, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            row = []
            toks = line.split()
            for tok in toks:
                try:
                    num = float(tok)
                except ValueError:

                    continue
                row.append(num)
            mat.append(row)
    # インスタンスの複数の行（問題設定）を取り出す
    Setting_Info = mat.pop(0)  # 0:車両数、4:キャパシティ、8:一台あたりの最大移動時間(min)、9:一人あたりの最大移動時間(min)

    # デポの座標を取り出す
    depo_zahyo = np.zeros(2)  # デポ座標配列
    x = mat.pop(-1)
    depo_zahyo[0] = x[1]
    depo_zahyo[1] = x[2]

    request_number = len(mat) - 1

    # 各距離の計算
    c = np.zeros((len(mat), len(mat)), dtype=float, order='C')

    # eがtime_windowの始、lが終
    e = np.zeros(len(mat), dtype=int, order='C')
    l = np.zeros(len(mat), dtype=int, order='C')

    # テキストファイルからtime_windowを格納 & 各ノードの距離を計算し格納
    for i in range(len(mat)):
        e[i] = mat[i][5]
        l[i] = mat[i][6]
        for j in range(len(mat)):
            c[i][j] = distance(mat[i][1], mat[j][1], mat[i][2], mat[j][2])

    # 乗り降りの0-1情報を格納
    noriori = np.zeros(len(mat), dtype=int, order='C')
    for i in range(len(mat)):
        noriori[i] = mat[i][4]

    return Setting_Info, request_number, depo_zahyo, c, e, l, noriori


def network_creat(Time_expand,kakucho):
    G = nx.Graph()  # ノード作成


    for i in range(n):
        early_time = e[i]
        late_time = l[i]
        if e[i] == 0:
            early_time = 0
        add_node = range(early_time, late_time)
        if i == 0:
            G.add_node((0, 0))
        else:
            for j in add_node:
                if j % Time_expand == 0:
                    G.add_node((i, j))
    G.add_node((n, T + 1))
    # G.add_edge((0,0),(1,5),weight=Setting_Info[3][0][1])

    for a in range(n):
        early_time = e[a]
        late_time = l[a]

        add_node = range(early_time, late_time)
        for j in add_node:
            if j % Time_expand == 0:
                b = 0
                for i in range(n - 1):  # 各ノードからdepoに帰るエッジがつくられていない & ここのループだとdepoのノード同士がつながらないので改善が必要
                    if a == 0 and noriori[i + 1] > 0:
                        next_early_time = e[i + 1]
                        next_late_time = l[i + 1]

                        next_add_node = range(next_early_time, next_late_time)
                        for k in next_add_node:
                            if k % Time_expand == 0:
                                distance_check = math.ceil(Distance[a][i + 1])
                                if distance_check + j <= k:  # このedgeを追加するコードは無駄な処理を含んでいます。直す必要アリ(5/10)
                                    b = 1
                                    if a == i + 1:
                                        if k - j == 1:
                                            G.add_edge((0, 0), (i + 1, k), weight=Distance[a][i + 1])
                                            G.edges[(0, 0), (i + 1, k)]['penalty'] = 0
                                            G.edges[(0, 0), (i + 1, k)]['ph'] = 1 / k

                                    else:
                                        G.add_edge((0, 0), (i + 1, k), weight=Distance[a][i + 1])
                                        G.edges[(0, 0), (i + 1, k)]['penalty'] = 0
                                        G.edges[(0, 0), (i + 1, k)]['ph'] = 1 / k

                                if b == 1:
                                    break
                    elif not a == 0 and not a - (i + 1) == Request:
                        if noriori[a] > 0:
                            next_early_time = e[i + 1]
                            next_late_time = l[i + 1]
                            connect_abs = l[a] - next_late_time
                            if abs(connect_abs) <= Setting_Info_base[9]:
                                next_add_node = range(next_early_time, next_late_time)
                                for k in next_add_node:
                                    if k > j:
                                        if k % Time_expand == 0:
                                            distance_check = math.ceil(Distance[a][i + 1])
                                            if distance_check + j <= k:  # このedgeを追加するコードは無駄な処理を含んでいます。直す必要アリ(5/10)
                                                b = 1
                                                if not a == i + 1:
                                                    G.add_edge((a, j), (i + 1, k), weight=Distance[a][i + 1])
                                                    G.edges[(a, j), (i + 1, k)]['penalty'] = 0
                                                    G.edges[(a, j), (i + 1, k)]['ph'] = 1 / abs(j - k)

                                            if b == 1:
                                                b = 0
                                                break
                        else:
                            next_early_time = e[i + 1]
                            next_late_time = l[i + 1]

                            next_add_node = range(next_early_time, next_late_time)
                            for k in next_add_node:
                                if k > j:
                                    if k % Time_expand == 0:
                                        distance_check = math.ceil(Distance[a][i + 1])
                                        if distance_check + j <= k:  # このedgeを追加するコードは無駄な処理を含んでいます。直す必要アリ(5/10)
                                            b = 1
                                            if not a == i + 1:
                                                G.add_edge((a, j), (i + 1, k), weight=Distance[a][i + 1])
                                                G.edges[(a, j), (i + 1, k)]['penalty'] = 0
                                                G.edges[(a, j), (i + 1, k)]['ph'] = 1 / abs(j - k)
                                        if b == 1:
                                            b = 0
                                            break

    for i in range(n - 1):
        if noriori[i + 1] < 0:
            early_time = e[i + 1]
            late_time = l[i + 1]

            add_node = range(early_time, late_time)
            for j in add_node:
                if j % Time_expand == 0:
                    b = 0
                    depo_repeat = range(early_time, l[0])
                    for k in depo_repeat:
                        if k % Time_expand == 0:
                            distance_check = math.ceil(Distance[i + 1][0])
                            if j + distance_check <= k:
                                b = 1
                                G.add_edge((i + 1, j), (n, T + 1), weight=Distance[i + 1][0])
                                G.edges[(i + 1, j), (n, T + 1)]['penalty'] = 0
                                G.edges[(i + 1, j), (n, T + 1)]['ph'] = 1 / (T + 1 - j)
                            if b == 1:
                                break
#以下ペナルティエッジの追加
    for a in range(n):
        early_time = e[a]
        late_time = l[a]

        add_node = range(early_time, late_time + Time_expand * kakucho)
        for j in add_node:
            if j % Time_expand == 0:
                b = 0
                for i in range(n - 1):  # 各ノードからdepoに帰るエッジがつくられていない & ここのループだとdepoのノード同士がつながらないので改善が必要

                    if not a == 0 and not a-(i+1) == Request:
                        if noriori[a] > 0:

                            next_late_time = l[i + 1]
                            connect_abs = l[a] - next_late_time
                            if abs(connect_abs) <= Setting_Info_base[9]:
                                next_add_node = range(next_late_time, next_late_time + Time_expand * kakucho)
                                for k in next_add_node:
                                    if k % Time_expand == 0:
                                        distance_check = math.ceil(Distance[a][i + 1])
                                        if distance_check + j <= k:  # このedgeを追加するコードは無駄な処理を含んでいます。直す必要アリ(5/10)
                                            b = 1
                                            if not a == i + 1:
                                                G.add_edge((a, j), (i + 1, k), weight=Distance[a][i + 1])
                                                G.edges[(a, j), (i + 1, k)]['penalty'] = 1
                                                G.edges[(a, j), (i + 1, k)]['ph'] = 1 / abs(j - k)
                                        if b == 1:
                                            b = 0
                                            break
                        else:
                            next_late_time = l[i + 1]

                            next_add_node = range(next_late_time, next_late_time + Time_expand * kakucho)
                            for k in next_add_node:
                                if k % Time_expand == 0:
                                    distance_check = math.ceil(Distance[a][i + 1])
                                    if distance_check + j <= k:  # このedgeを追加するコードは無駄な処理を含んでいます。直す必要アリ(5/10)
                                        b = 1
                                        if not a == i + 1:
                                            G.add_edge((a, j), (i + 1, k), weight=Distance[a][i + 1])
                                            G.edges[(a, j), (i + 1, k)]['penalty'] = 1
                                            G.edges[(a, j), (i + 1, k)]['ph'] = 1 / abs(j - k)
                                    if b == 1:
                                        b = 0
                                        break
        for i in range(n - 1):
            if noriori[i + 1] < 0:
                early_time = e[i + 1]
                late_time = l[i + 1]

                add_node = range(late_time, late_time + Time_expand * kakucho)
                for j in add_node:
                    if j % Time_expand == 0:
                        b = 0
                        depo_repeat = range(l[0], l[0] + Time_expand * kakucho)
                        for k in depo_repeat:
                            if k % Time_expand == 0:
                                distance_check = math.ceil(Distance[i + 1][0])
                                if j + distance_check <= k:
                                    b = 1
                                    G.add_edge((i + 1, j), (n, T + 1), weight=Distance[i + 1][0])
                                    G.edges[(i + 1, j), (n, T + 1)]['penalty'] = 1
                                    G.edges[(i+1,j),(n,T+1)]['ph'] = 1/(T+1-j)
                                if b == 1:
                                    break


    pos = {n: (n[1], -n[0]) for n in G.nodes()}  # ノードの座標に注意：X座標がノード番号、Y座標が時刻t

    c_edge = ['red' if G.edges[(n)]['penalty'] == 1 else 'black' for n in G.edges()]

    return G

def check_node(next_location_id):
    flag = 1
    next_location_id = genzaichi_update(next_location_id)
    try:
        next_location_dic = G.adj[next_location_id]
    except KeyError as e:
        flag = 0

    return flag

def genzaichi_update(tup):
    tup_new = list(tup)
    tup_new[1] = tup_new[1] + d
    return tuple(tup_new)

def syaryo_time_check(Loot):
    syaryo_time = 0
    if not Loot == []:
        syaryo_time = Loot[-1][1] + Distance[0][Loot[-1][0]] - (Loot[0][1] - Distance[0][Loot[0][0]])
    return syaryo_time

def sharing_number_count(Loot):
    number_count = 0
    for i in Loot:
        number_count += noriori[i]

    return number_count

def update_pick_node(next_node, pick_list):
    if noriori[next_node[0]] == 1:
        pick_list.append(next_node[0] + Request)
    else:
        pick_list.remove(next_node[0])

    return pick_list

"""
関数network_updateについて
現在地のノードを削除したらいけません→この関数は使えないかも
一台分のルートが完成してから削除しましょう
"""

def network_update(network, removenode):
    for i in list(network.nodes()):
        for j in removenode:
            if i[0] == j:
                network.remove_node(i)

'''
return_random関数に関して
内容：接続可能なノード一覧からランダムで移動するノード（タプル型）を返す関数
備考：接続ノード番号をランダムでチョイスしてその中で最も時間が早いノードを返している
これから：①ピックアップノードが現在地として入力されたとき、乗車客が定員以下の場合は、ピックアップノード+これまで乗せた乗車客のドロップノードの中から選択
        ②ピックアップノードが現在地かつ乗車客が定員Maxの場合は、乗せている乗車客のドロップノードから選択⇒ここはランダムで選択しなくて良い、最も締め切り時間に近いのを選択
        ③ドロップノードが現在地のとき、ピックアップノードor今乗せている乗車客のドロップノードの中から選択
        例外処理：移動可能なノードがない場合デポを返す
'''

def drop_check(picking_list, next_location):
    flag = 0
    if not picking_list == []:
        dic = G.adj[next_location]
        for id, info in dic.items():
            if id[0] in picking_list:
                flag += 1
    else:
        flag += 1

    return flag

def return_random(dic, now_location, capacity, picking_list):
    idou_kanou = []
    idou_list = []
    next_limit = Setting_Info_base[9]
    saitan_drop_node = (n, T + 1)
    random_return = (0, 0)
    if noriori[now_location[0]] == 1 or noriori[now_location[0]] == 0:
        if noriori[now_location[0]] == 0:
            for id, info in dic.items():
                if id[1] > now_location[1] and not id[0] == n and check_node(id) == 1:
                    if noriori[id[0]] == 1:
                        idou_kanou.append(id)
                        idou_list.append(id[0])
                        if id[1] < saitan_drop_node[1]:
                            saitan_drop_node = id

            random_return = saitan_drop_node
            if saitan_drop_node == (10000, 10000):
                random_return = (n, T + 1)
        else:
            for id, info in dic.items():
                if id[1] > now_location[1] and id[1] < now_location[1] + next_limit and not id[0] == n and id[
                    0] not in kanryo_node and check_node(id):
                    if noriori[id[0]] == 1:
                        idou_kanou.append(id)
                        idou_list.append(id[0])
                    else:
                        if id[0] in picking_list:
                            idou_kanou.append(id)
                            idou_list.append(id[0])
                            if id[1] < saitan_drop_node[1]:
                                saitan_drop_node = id

            if not idou_kanou == []:
                randam = random.choice(list(set(idou_list)))

                idou_kanou = np.array(idou_kanou)
                random_return = tuple(idou_kanou[np.any(idou_kanou == randam, axis=1)][0])
                if random_return[1] > saitan_drop_node[1] or drop_check(picking_list, random_return) == 0:
                    random_return = saitan_drop_node

            else:
                random_return = (n, T + 1)

    else:
        for id, info in dic.items():
            if id[1] > now_location[1] and id[1] < now_location[1] + next_limit * 1.5 and id[
                0] not in kanryo_node and check_node(id) == 1:
                if noriori[id[0]] == 1:
                    idou_kanou.append(id)
                    idou_list.append(id[0])
                else:
                    if id[0] in picking_list:
                        idou_kanou.append(id)
                        idou_list.append(id[0])
                        if id[1] < saitan_drop_node[1]:
                            saitan_drop_node = id

        if not idou_kanou == []:
            randam = random.choice(list(set(idou_list)))

            idou_kanou = np.array(idou_kanou)
            random_return = tuple(idou_kanou[np.any(idou_kanou == randam, axis=1)][0])
            if random_return[1] > saitan_drop_node[1] or drop_check(picking_list, random_return) == 0:
                random_return = saitan_drop_node

        else:
            random_return = (n, T + 1)

    return random_return

def return_kakuritsu(dic, now_location, capacity, picking_list):
    idou_kanou = []
    idou_kanou_time = []
    idou_kakuritsu = []
    next_limit = Setting_Info_base[9]
    capa_max = Setting_Info_base[4]
    saitan_drop_node = (n, T + 1)
    random_return = (0, 0)
    if capacity < capa_max:
        if noriori[now_location[0]] == 0:
            for id, info in dic.items():
                if not id[0] == n and check_node(id) == 1:
                    if noriori[id[0]] == 1:
                        if id[0] in idou_kanou:
                            break
                        idou_kanou.append(id[0])
                        idou_kanou_time.append(id[1])
                        idou_kakuritsu.append(list(info.values())[2])
            if not idou_kanou == []:
                random_return = saisyo(idou_kanou[idou_kanou_time.index(min(idou_kanou_time))],
                                       min(idou_kanou_time))

        elif noriori[now_location[0]] == 1:
            for id, info in dic.items():
                if id[1] < now_location[1] + next_limit and not id[0] == n and id[
                    0] not in kanryo_node and check_node(id):
                    if id[0] in idou_kanou:
                        break
                    if noriori[id[0]] == 1:
                        idou_kanou.append(id[0])
                        idou_kanou_time.append(id[1])
                        idou_kakuritsu.append(list(info.values())[2])
                    else:
                        if id[0] in picking_list:
                            idou_kanou.append(id[0])
                            idou_kanou_time.append(id[1])
                            idou_kakuritsu.append(list(info.values())[2])
            random_return = probability_choice(now_location, idou_kanou, idou_kakuritsu, idou_kanou_time)
        elif noriori[now_location[0]] == -1:
            for id, info in dic.items():
                if not picking_list == []:
                    if id[1] < now_location[1] + next_limit and not id[0] == n and id[
                        0] not in kanryo_node and check_node(id):
                        if id[0] in idou_kanou:
                            break
                        if noriori[id[0]] == 1:
                            idou_kanou.append(id[0])
                            idou_kanou_time.append(id[1])
                            idou_kakuritsu.append(list(info.values())[2])
                        else:
                            if id[0] in picking_list:
                                idou_kanou.append(id[0])
                                idou_kanou_time.append(id[1])
                                idou_kakuritsu.append(list(info.values())[2])
                else:
                    if id[0] not in kanryo_node and check_node(id):
                        if id[0] in idou_kanou:
                            break
                        if noriori[id[0]] == 1:
                            idou_kanou.append(id[0])
                            idou_kanou_time.append(id[1])
                            idou_kakuritsu.append(list(info.values())[2])
                        else:
                            if id[0] in picking_list:
                                idou_kanou.append(id[0])
                                idou_kanou_time.append(id[1])
                                idou_kakuritsu.append(list(info.values())[2])
            random_return = probability_choice(now_location, idou_kanou, idou_kakuritsu, idou_kanou_time)
    else:
        pass

    if random_return == (0, 0):
        random_return = (n, T + 1)
    return random_return

def saisyo(saisyo_kyaku, saisyo_time):
    re_saisyo = [saisyo_kyaku, saisyo_time]
    re_saisyo = tuple(re_saisyo)
    return re_saisyo

def total_distance(loot):
    Total = np.zeros(len(loot))

    for i in range(len(loot)):
        if not loot[i] == []:
            kyori = Distance[loot[i][0][0]][0] + Distance[loot[i][-1][0]][n - 1]
            Total[i] += kyori
            for j in range(len(loot[i]) - 1):
                kyori = Distance[loot[i][j][0]][loot[i][j + 1][0]]
                Total[i] += kyori
    return Total

def daisu_check(loot):
    number = 0
    for i in range(len(loot)):
        if not len(loot[i]) == 0:
            number += 1
    return number

def probability_choice(now_location, idou_list, idou_probability, idou_kanou_time):
    if not idou_list == []:
        re_random = []
        kakuritsu_list = cal_kakuritsu(now_location, idou_list, pheromon=idou_probability)

        random = np.random.choice(idou_list, p=kakuritsu_list)
        index = idou_list.index(random)
        re_random.append(random)
        re_random.append(idou_kanou_time[index])
        re_random = tuple(re_random)
    else:
        re_random = (n, T + 1)
    return re_random

def cal_kakuritsu(now_location, idou_list, pheromon):
    kakuritsu_list = []
    sum = 0
    sum_sum = 0
    for i in range(len(pheromon)):
        if noriori[idou_list[i]] == -1:
            p = (pheromon[i] ** alpha) * ((Q / Distance[idou_list[i]][now_location[0]]) ** beta) + 1 / (
                        l[idou_list[i]] - e[idou_list[i]])
            kakuritsu_list.append(p)
            sum += p
        else:
            p = (pheromon[i] ** alpha) * ((Q / Distance[idou_list[i]][now_location[0]]) ** beta)
            kakuritsu_list.append(p)
            sum += p
    for i in range(len(kakuritsu_list)):
        if i == 0:
            kakuritsu_list[i] = kakuritsu_list[i] / sum
            sum_sum += kakuritsu_list[i]
        elif i == range(len(kakuritsu_list)):
            kakuritsu_list[i] = 1 - sum_sum
        else:
            kakuritsu_list[i] = kakuritsu_list[i] / sum
            sum_sum += kakuritsu_list[i]

    return kakuritsu_list

if __name__ == '__main__':
    t1=time.time()
    FILENAME = 'darp01EX.txt'
    Setting_Info = Setting(FILENAME)
    Setting_Info_base = Setting_Info[0]

    Syaryo_max_time = Setting_Info_base[8]
    T = int(Setting_Info_base[5])  # 時間数
    n = int(Setting_Info[1]) + 1  # デポを含めた頂点数
    Request = int((n - 1) / 2)  # リクエスト数
    Distance = Setting_Info[3]  # 距離
    e = Setting_Info[4]  # early time
    l = Setting_Info[5]  # delay time
    d = 5  # 乗り降りにようする時間
    noriori = Setting_Info[6]

    time_expand = 1

    G = network_creat(Time_expand=time_expand, kakucho=30)

    G_copy = copy.deepcopy(G)

    alpha = 1
    beta = 1
    Q = 1
    print(FILENAME)
    print(time_expand)
    print(nx.number_of_edges(G))
    print(nx.number_of_nodes(G))

    genzaichi = (0, 0)
    old_genzaichi = genzaichi
    print(G.adj[genzaichi])
    # print(G.adj[genzaichi][(1, 5)].values())
    print(G.adj[genzaichi].values())
    print(type(G.adj[genzaichi]))
    roop = 0
    data = np.zeros((500, 2))
    opt = 10000
    opt_loot = []
    misounyu = []
    misounyu_2 = []
    t3 = time.time()
    while True:
        G = copy.deepcopy(G_copy)
        main_loop = 0

        loot = [[] * 1 for i in range(10)]

        kanryo_node = []
        pick_now_node_list = []
        while True:
            genzaichi = (0, 0)
            old_genzaichi = genzaichi
            capa = 0
            pick_now_node_list = []
            while True:
                setuzoku_Node = return_kakuritsu(G.adj[genzaichi], genzaichi, capa, pick_now_node_list)
                if not setuzoku_Node[0] == n:
                    pick_now_node_list = update_pick_node(setuzoku_Node, pick_now_node_list)
                    if noriori[setuzoku_Node[0]] == 1:
                        capa += 1
                    else:
                        capa -= 1
                if pick_now_node_list == [] and syaryo_time_check(loot[main_loop]) >= Syaryo_max_time:
                    break
                if setuzoku_Node == (n, T + 1):
                    break

                kanryo_node.append(setuzoku_Node[0])

                old_genzaichi = genzaichi
                genzaichi = setuzoku_Node

                loot[main_loop].append(genzaichi)

                genzaichi = genzaichi_update(genzaichi)
                loot[main_loop].append(genzaichi)

                # if main_loop == 3:
                #    break

            # print(loot[main_loop])
            # print(loot)
            # print(syaryo_time_check(loot[main_loop]))
            # print(kanryo_node)
            network_update(G, kanryo_node)
            main_loop += 1
            misounyu_2.append(kanryo_node)
            kanryo_node = []
            misounyu.append(pick_now_node_list)
            if main_loop == 3:
                break
        roop += 1
        if roop == 1:
            break
    t2=time.time()
    print(f"time:{t2-t1}")
    print(f"time{t2-t3}")
    print(loot)
    print(total_distance(opt_loot))
    print(misounyu)
    syaryo = 0
    for i in range(len(loot)):
        if not loot[i] == []:
            syaryo += 1
    print(syaryo)
    kokyaku_node = range(1, 49)
    print(sum(misounyu_2, []))
    print(set(kokyaku_node) ^ set(sum(misounyu_2, [])))
    #np.savetxt('/Users/kurozumi ryouho/Desktop/benchmark2/kekka/' + FILENAME + 'ans.csv', data, delimiter=",")