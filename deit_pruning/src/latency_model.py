import json 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import shutil,json
def get_accuracy(y_pred,y_true,threshold=0.01):
    a=(y_true-y_pred)/y_true 
    c=abs(y_true-y_pred)

    b=(np.where(abs(a)<=threshold ) )
    return len(b[0])/len(y_true)
   


def lat_metrics(y_pred,y_true):
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    rmse=np.sqrt(mean_squared_error(y_pred,y_true))    
    acc5=get_accuracy(y_pred,y_true,threshold=0.05)
    acc10=get_accuracy(y_pred,y_true,threshold=0.10)
    acc15=get_accuracy(y_pred,y_true,threshold=0.15)
   

    return rmse,rmspe,rmse/np.mean(y_true),acc5,acc10,acc15
def get_feature(fe):
    layers=fe.split('-')
    X=[]
    for layer in layers:
        items=layer.split('_')
        h=float(items[1])
        d=float(items[-1])
        X.append(h)
        X.append(d)
    return X


def get_latency(filename):
    X=[]
    Y=[]
    f1=open("latency.csv",'w')
    with open(filename,'r') as fw:
        dicts=json.load(fw)
        for mid in dicts:
            #print(mid,dicts[mid])
            fe=mid.split('\\')[-1].replace(".onnx","")
            data=dicts[mid]
            items=data.split('\r\n')
           # print(items)
            avg=float(items[-3].split(': ')[-1].replace(" us",""))
            x=get_feature(fe)
            print(fe,avg)
            X.append(x)
            Y.append(avg)
            f1.write(fe+','+str(avg)+'\n')
    return X,Y

def get_model(filename):
    X,Y=get_latency(filename)
    print(len(X))
    trainx, testx, trainy, testy = train_test_split(
                    X, Y, test_size=0.2, random_state=10
                )

    print(min(Y),max(Y),np.average(Y))


    model=RandomForestRegressor(max_depth=70,n_estimators=320,min_samples_leaf=1,min_samples_split=2,
                                            max_features=8, oob_score=True,random_state=10)

    model.fit(trainx,trainy)
    predicts=model.predict(testx)
    rmse,rmspe,error,acc5,acc10,_=lat_metrics(predicts,testy)
    print(rmse,rmspe,error,acc5,acc10)
    for i in range(len(testy)):
        print(testy[i],predicts[i],(testy[i]-predicts[i])/testy[i])

    model.fit(X, Y)
    with open("latency/latency_model.pkl", "wb") as f:
                    pickle.dump(model, f)

def predict(feature):
    with open('latency/latency_model.pkl', "rb") as f: 
        model = pickle.load(f)
        return model.predict(feature)[0]

#get_model("latency/latency_bench_newt13.json")
fe=get_feature('h_4_d_0.4-h_4_d_0.4-h_4_d_0.4-h_4_d_0.4')
print(fe)
latency=predict([fe])

print(latency)