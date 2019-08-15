from underthesea import ner
import pandas as pd 

def getLOC(message_text):
    filename = "data/dataset_XY_XLS_updatedbytho_ver2_2.xlsx"
    df = pd.read_excel(filename, sheet_name="listofWorkingplace", encoding="utf8")

    # use underthesea tool: ner to N-E-R sentence
    res = ner(message_text)
    
    # filter the result to get the I-LOC or B-LOC
    loc_VN = list(df.loc[:,"Name"])
    loc_stop = ['huyện', 'tỉnh', 'thị xã', 'quận', 'khu phố', 'phường', 'ấp', 'thành phố', 'thành phố mới']
    locIDX = []
    for i in range (len(res)):
        if res[i][3] == "I-LOC" or res[i][3] == "B-LOC":
            if res[i][0] not in loc_stop and res[i][0] in loc_VN:
                locIDX.append(i)
    # dict-orize to transform to value of Rasa
    dictLOC = {}
    count = []
    countLOC = []
    for i in range(len(locIDX)):
        count.append("LOC "+ str(i))
    for i in locIDX:
        countLOC.append(res[i][0])
    dictLOC = dict(zip(count, countLOC))

    return dictLOC

def getNerName(message_text):
    filename = "data/dataset_XY_XLS_updatedbytho_ver2_2.xlsx"
    df = pd.read_excel(filename, sheet_name="listofWorkingplace", encoding="utf8")

    # use underthesea tool: ner to N-E-R sentence
    res = ner(message_text)
    
    # filter the result to get the I-LOC or B-LOC
    loc_VN = list(df.loc[:,"Name"])
    loc_stop = ['huyện', 'tỉnh', 'thị xã', 'quận', 'khu phố', 'phường', 'ấp', 'thành phố', 'thành phố mới']
    locIDX = []
    locNAME=[]
    res = ner(message_text)
    print('res: ', res)
    for i in range (len(res)):
        if res[i][0] not in loc_stop and res[i][0] in loc_VN:
            if res[i][3] == "I-LOC" or res[i][3] == "B-LOC":
                locIDX.append(i)
    for i in locIDX:
        locNAME.append(res[i][0])
    return locIDX, locNAME

def isAnyLOC(message_text):
    isLOC = False
    res = ner(message_text)
    for i in range (len(res)):
        if res[i][3] == "I-LOC" or res[i][3] == "B-LOC":
            isLOC = True
    return isLOC
