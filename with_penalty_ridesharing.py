import numpy as np
import pandas as pd
import networkx as nx
import math
from itertools import product
import matplotlib.pyplot as plt
import random


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

def setuzoku_node_list(dic):  # 移動できるノードの一覧辞書を返す関数、時間軸に関しては情報落ち
    node_dict = {}
    for id, info in dic.items():
        print(id, info.values())
        node_dict.setdefault(id[0], info.values())

    return node_dict


"""
関数setuzoku_nodeについて
二個目のノードから再び(0,0)のノードが選ばれてしまうので何かしら分岐が必要
・ピックアップノードからデポに戻るの禁止
    ＊デポ以外の近いところを選択、どうしてもない場合、ドロップノードに行く
・6/20
    ＊ピックアップノードから関係ないドロップノードにいってしまう
    *
・6/21 
    *方向性を転換⇒時間窓の早い顧客から詰め込む
"""


def setuzoku_node_list2(dic, now_location, previous_location):  #dic⇒接続可能なノード先(G.adj[n])
    min_weight = float('inf')
    min_earlytime = float('inf')
    saitan_setuzoku_node = (0, 0)
    drop_kouho = (0, 0)
    loop = 0
    if genzaichi == (0, 0):
        for id, info in dic.items():
            # print(id, info.values())
            if loop == 0:
                saitan_setuzoku_node = id
                min_earlytime = id[1]
            else:
                if id[1] < min_earlytime:
                    saitan_setuzoku_node = id
                    min_earlytime = id[1]

            loop += 1
    elif noriori[now_location[0]] == 1:  # 現在地がピックアップノードのとき
        for id, info in dic.items():
            # print(id, info.values())
            if id[0] == now_location[0] + Request:
                drop_kouho = id
                saitan_setuzoku_node = drop_kouho
                break
            loop += 1


    elif noriori[now_location[0]] == -1:
        for id, info in dic.items():
            # print(id, info.values())
            if loop == 0:
                if not id[0] == n:
                    if not id == previous_location and noriori[id[0]] == 1 and (id[0] in kanryo_node) == False and id[
                        1] > now_location[1]:   #候補が以前の場所ではない & ピックアップノードである & 　挿入済みのノードではない & 現在の時間より大きいノードである
                        if check_node(id, now_location) == 1:
                            saitan_setuzoku_node = id
                            min_earlytime = id[1]
                else:
                    break
            else:
                if id[0] == n:
                    if not id == previous_location and noriori[0] == 1:
                        if id[1] < min_earlytime:
                            saitan_setuzoku_node = id
                            min_earlytime = id[1]
                else:
                    if not id == previous_location and noriori[id[0]] == 1 and (id[0] in kanryo_node) == False and id[
                        1] > now_location[1]:
                        if check_node(id, now_location) == 1:
                            if id[1] < min_earlytime:
                                saitan_setuzoku_node = id
                                min_earlytime = id[1]
            loop += 1
    if saitan_setuzoku_node == (0, 0):
        saitan_setuzoku_node = (n, T + 1)
    return saitan_setuzoku_node


"""
#別のピックアップノードを入れたあと、以前のピックアップをドロップできるか判定する
"""


def check_node(next_location_id, now_location):
    flag = 0
    next_location_id = genzaichi_update(next_location_id)
    try:
        next_location_dic = G.adj[next_location_id]
        for id, info in next_location_dic.items():
            if id[0] == next_location_id[0] + Request:
                flag = 1
                break
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
    number_count=0
    for i in Loot:
        number_count+=noriori[i]

    return  number_count

def update_pick_node(next_node,pick_list):
    if noriori[next_node[0]] ==1:
        pick_list.append(next_node[0]+Request)
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


def return_random(dic, now_location,capacity,picking_list):
    idou_kanou = []
    idou_list = []
    if noriori[now_location[0]] == 1 or noriori[now_location[0]] ==0:
        for id, info in dic.items():
            if id[1] > now_location[1] and not id[0] == n:
                if noriori[id[0]] ==1:
                    idou_kanou.append(id)
                    idou_list.append(id[0])
                else:
                    if id[0] in  picking_list:
                        idou_kanou.append(id)
                        idou_list.append(id[0])

        print(idou_kanou)
        print(idou_list)
        if not idou_kanou == []:
            randam = random.choice(list(set(idou_list)))
            print(randam)
            idou_kanou = np.array(idou_kanou)
            random_return = tuple(idou_kanou[np.any(idou_kanou == randam, axis=1)][0])
            print(random_return)
        else:
            random_return = (n, T + 1)
    else:
        for id, info in dic.items():
            if id[1] > now_location[1] and not id[0] == n:
                idou_kanou.append(id)
                idou_list.append(id[0])

        print(idou_kanou)
        print(idou_list)
        if not idou_kanou == []:
            randam = random.choice(list(set(idou_list)))
            print(randam)
            idou_kanou = np.array(idou_kanou)
            random_return = tuple(idou_kanou[np.any(idou_kanou == randam, axis=1)][0])
            print(random_return)
        else:
            random_return = (n, T + 1)
    return random_return


def return_saitan(dic, now_location,capacity,picking_list):
    idou_kanou = []
    idou_list = []
    if noriori[now_location[0]] == 1 or noriori[now_location[0]] ==0:
        for id, info in dic.items():
            if id[1] > now_location[1] and not id[0] == n:
                if noriori[id[0]] ==1:
                    idou_kanou.append(id)
                    idou_list.append(id[0])
                else:
                    if id[0] in  picking_list:
                        idou_kanou.append(id)
                        idou_list.append(id[0])

        print(idou_kanou)
        print(idou_list)
        if not idou_kanou == []:
            randam = random.choice(list(set(idou_list)))
            print(randam)
            idou_kanou = np.array(idou_kanou)
            random_return = tuple(idou_kanou[np.any(idou_kanou == randam, axis=1)][0])
            print(random_return)
        else:
            random_return = (n, T + 1)
    else:
        for id, info in dic.items():
            if id[1] > now_location[1] and not id[0] == n:
                idou_kanou.append(id)
                idou_list.append(id[0])

        print(idou_kanou)
        print(idou_list)
        if not idou_kanou == []:
            randam = random.choice(list(set(idou_list)))
            print(randam)
            idou_kanou = np.array(idou_kanou)
            random_return = tuple(idou_kanou[np.any(idou_kanou == randam, axis=1)][0])
            print(random_return)
        else:
            random_return = (n, T + 1)
    return random_return

if __name__ == '__main__':
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
    d = 10  # 乗り降りにようする時間
    noriori = Setting_Info[6]

    Time_expand = 1

    kakucho =40

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
                                    else:
                                        G.add_edge((0, 0), (i + 1, k), weight=Distance[a][i + 1])
                                        G.edges[(0, 0), (i + 1, k)]['penalty'] = 0

                                if b == 1:
                                    break
                    elif not a == 0:
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
                                            G.add_edge((a, j), (i + 1, k), weight=Distance[a][i + 1])
                                            G.edges[(a, j), (i + 1, k)]['penalty'] = 0
                                    else:
                                        G.add_edge((a, j), (i + 1, k), weight=Distance[a][i + 1])
                                        G.edges[(a, j), (i + 1, k)]['penalty'] = 0
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
                                G.edges[(i + 1, j), (n, T + 1)]['penalty'] =0
                            if b == 1:
                                break

    for a in range(n):
        early_time = e[a]
        late_time = l[a]

        add_node = range(early_time, late_time+Time_expand*kakucho)
        for j in add_node:
            if j % Time_expand == 0:
                b = 0
                for i in range(n - 1):  # 各ノードからdepoに帰るエッジがつくられていない & ここのループだとdepoのノード同士がつながらないので改善が必要
                    if a == 0 and noriori[i + 1] > 0:
                        next_late_time = l[i + 1]

                        next_add_node = range(next_late_time,next_late_time+Time_expand*kakucho)
                        for k in next_add_node:
                            if k % Time_expand == 0:
                                distance_check = math.ceil(Distance[a][i + 1])
                                if distance_check + j <= k:  # このedgeを追加するコードは無駄な処理を含んでいます。直す必要アリ(5/10)
                                    b = 1
                                    if a == i + 1:
                                        if k - j == 1:
                                            G.add_edge((0, 0), (i + 1, k), weight=Distance[a][i + 1])
                                            G.edges[(0, 0), (i + 1, k)]['penalty'] = 1
                                    else:
                                        G.add_edge((0, 0), (i + 1, k), weight=Distance[a][i + 1])
                                        G.edges[(0, 0), (i + 1, k)]['penalty'] = 1
                                if b == 1:
                                    break
                    elif not a == 0:
                        next_late_time = l[i + 1]

                        next_add_node = range(next_late_time,next_late_time+Time_expand*kakucho)
                        for k in next_add_node:
                            if k % Time_expand == 0:
                                distance_check = math.ceil(Distance[a][i + 1])
                                if distance_check + j <= k:  # このedgeを追加するコードは無駄な処理を含んでいます。直す必要アリ(5/10)
                                    b = 1
                                    if not a == i+1:
                                        G.add_edge((a, j), (i + 1, k), weight=Distance[a][i + 1])
                                        G.edges[(a, j), (i + 1, k)]['penalty'] = 1
                                if b == 1:
                                    b=0
                                    break
        for i in range(n - 1):
            if noriori[i + 1] < 0:
                early_time = e[i + 1]
                late_time = l[i + 1]

                add_node = range(early_time, late_time+Time_expand*kakucho)
                for j in add_node:
                    if j % Time_expand == 0:
                        b = 0
                        depo_repeat = range(l[0], l[0]+Time_expand*kakucho)
                        for k in depo_repeat:
                            if k % Time_expand == 0:
                                distance_check = math.ceil(Distance[i + 1][0])
                                if j + distance_check <= k:
                                    b = 1
                                    G.add_edge((i + 1, j), (n, T+1), weight=Distance[i + 1][0])
                                    G.edges[(i + 1, j), (n, T+1)]['penalty'] = 1
                                if b == 1:
                                    break

    pos = {n: (n[1], -n[0]) for n in G.nodes()}  # ノードの座標に注意：X座標がノード番号、Y座標が時刻t

    c_edge = ['red' if G.edges[(n)]['penalty'] == 1 else 'black' for n in G.edges()]

    print(FILENAME)
    print(Time_expand)
    print(nx.number_of_edges(G))
    print(nx.number_of_nodes(G))

    genzaichi = (0, 0)
    old_genzaichi = genzaichi
    print(G.adj[genzaichi])
    # print(G.adj[genzaichi][(1, 5)].values())
    print(G.adj[genzaichi].values())
    print(type(G.adj[genzaichi]))
    main_loop = 0

    loot = [[] * 1 for i in range(6)]
    print(loot)

    kanryo_node = []
    pick_now_node_list =[]


    #test = return_random(G.adj[(47, 589)], (47, 589))
    #print(test)

    ru_to =[2,3,5]

    print(sharing_number_count(ru_to))

    while True:
        genzaichi = (0, 0)
        old_genzaichi = genzaichi
        capa =0
        while True:
            setuzoku_Node = return_random(G.adj[genzaichi], genzaichi,capa,pick_now_node_list)
            if not setuzoku_Node[0] ==n:
                pick_now_node_list = update_pick_node(setuzoku_Node,pick_now_node_list)
                if noriori[setuzoku_Node[0]] ==1:
                    capa +=1
                else:
                    capa-=1
            '''
                if not setuzoku_Node == (n,T+1) and noriori[setuzoku_Node[0]] ==-1 and syaryo_time_check(loot[0]) >Syaryo_max_time:
                loot[0].pop()
                loot[0].pop()
                break    
            '''

            if setuzoku_Node == (n, T + 1):
                while True:
                    if syaryo_time_check(loot[main_loop]) > Syaryo_max_time:
                        loot[main_loop].pop()
                        loot[main_loop].pop()
                        kanryo_node.pop()
                        loot[main_loop].pop()
                        loot[main_loop].pop()
                        kanryo_node.pop()
                    else:
                        break
                break

            kanryo_node.append(setuzoku_Node[0])

            old_genzaichi = genzaichi
            genzaichi = setuzoku_Node

            loot[main_loop].append(genzaichi)

            genzaichi = genzaichi_update(genzaichi)
            loot[main_loop].append(genzaichi)

            # if main_loop == 3:
            #    break

        print(loot[main_loop])
        print(loot)
        print(syaryo_time_check(loot[main_loop]))
        print(kanryo_node)
        network_update(G, kanryo_node)
        main_loop += 1
        kanryo_node = []
        if main_loop == len(loot):
            break