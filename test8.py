import pickle
with open("D:\diao\Keyan\IBSR_Pytorch\data\pix3d\\rendering_pix3d.pkl",'rb') as f:
    data = pickle.load(f)
for key1 in data.keys():
    for key2 in data[key1].keys():
        print(data[key1][key2])
    break