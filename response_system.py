from intent_module_3 import *
from ner_module import *
import pandas as pd
from keras import backend as K


def getLocApply(locName):        
    filename='data/dataset_XY_XLS_updatedbytho_ver2_2.xlsx'
    ddata = pd.read_excel(filename ,sheet_name='listofWorkingplace')
    info = []
    rep = []

    for idx, iName in enumerate(ddata['Name']):
        if locName == iName:
            info = ddata.loc[idx,'Address']
    if info == [] and locName == []:
        rep = '\nKhông tìm thấy địa chỉ của bạn.'
    else:
        rep = '\nNếu bạn đang ở {0} thì bạn có thể đến {1}'.format(locName, info)
        rep = rep + '\nBạn có thể liên hệ với phòng Quản lý XNC {0} qua sdt +84{1}'.format(locName, ddata.loc[idx,'Number Phone'])
    return rep

def normalRep(intent):
    filename='data/dataset_XY_XLS_updatedbytho_ver2_2.xlsx'
    ddata = pd.read_excel(filename ,sheet_name='response_list')

    intent_ls = list(ddata.loc[:,'Tag'])
    rep = []
    for idx, inte in enumerate(intent_ls):
        if intent == inte:
            rep = ddata.loc[idx, 'Response']
            break
    
    return rep

def repOfficial(message_text):
    bot_rep = []
    
    userIntent = ic_predict(message_text)[0]
    # K.clear_session()
    print("user's intent: ", userIntent)

    if userIntent == "where_loc_apply":
        if isAnyLOC(message_text) == False:
            text = input('Bạn đang ở tỉnh thành: ')
            locQuantity, locApply = getNerName('Làm hộ chiếu ở ' + text)
            print(locApply)
            for i in range(len(locQuantity)):
                bot_rep.append(getLocApply(locApply[i]))
        else:
            locQuantity, locApply = getNerName(message_text)
            print(locApply)
            if locApply == []:
                bot_rep = getLocApply(locApply)
            else:
                for i in range(len(locQuantity)):
                    bot_rep.append(getLocApply(locApply[i]))
    else:
        bot_rep = normalRep(userIntent)
        ## Viết thêm cái điều kiện kiểm tra coi bot_rep là str hay list để xử lý xuất cho hiệu quả.
    str_bot_rep = ''
    for i in range(len(bot_rep)):
        str_bot_rep = str_bot_rep + bot_rep[i]
    return str_bot_rep
     

# while True:    
#     message_text = input('User say: ')


#     if message_text == 'stop':
#         break
#     response = repOfficial(message_text)
#     print('type of response: ', type(response))
#     print('Bot say: ', response)

# message_text = ' Tôi muốn đăng ký hộ chiếu ở Đồng Nai và Gia Lai'
# response = repOfficial(message_text)
# print(response)
