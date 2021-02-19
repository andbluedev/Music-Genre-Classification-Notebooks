from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb
import numpy as np
import librosa
import csv


##########################################
# Analysis
##########################################
def show_pca(X, y, title="", k_clusters_start=5):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    le = preprocessing.LabelEncoder()
    le.fit(y)

    y_encoded = le.transform(y)

    fig, ax = plt.subplots(1, figsize=(30,20))
    
    fig.suptitle(title, fontsize=30)
    scatter0 = ax.scatter(X_pca[:,0], X_pca[:,1], c=y_encoded, s=30,cmap=plt.get_cmap("Paired"))

    # Produce a legend with the unique colors from the scatter
    legend0 = ax.legend(*scatter0.legend_elements(),
                        loc="lower left", title="Classes")

    # Convert legend text
    for i in range(len(legend0.get_texts())):
        legend0.get_texts()[i].set_text(str(le.inverse_transform([i])[0]))
    
    ax.add_artist(legend0)
    ax.set_title("Plot of the PCA with dataset classes")

    plt.show()


def plot_histograms(df, feature_name):
    fig, ax = plt.subplots(figsize=(30,20))
    df[feature_name].hist(by=df['label'], figsize=(20,12), ax=ax)
    fig.suptitle("%s histogram by class" % feature_name, fontsize=30)
    plt.show()


def plot_boxes(df, feature_name):
    fig, ax = plt.subplots(figsize=(15,10))
    plt.suptitle("%s boxplot by class" % feature_name, fontsize=20)
    df.boxplot(column=[feature_name], by='label', ax=ax)
    plt.show()

def show_confusion_matrix(y_true, y_pred, encoder = None, title=""):

    if encoder is not None:
        for number, classname in enumerate(encoder.classes_):
            print(f"Class {number}: {classname}")

    fig, ax = plt.subplots(1, figsize=(10,10))
    
    fig.suptitle(title, fontsize=30)
    cf_matrix = confusion_matrix(y_true, y_pred)
    sb.heatmap(cf_matrix, annot=True, ax=ax)
    plt.show()






##########################################
# Predictions
##########################################
def predict_genre(model, scaler, songname, encoder = None):
    y, sr = librosa.load(songname, mono=True, offset = 60 , duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    #added 10/02
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr = sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr = sr)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(y=y)
    tonnetz = librosa.feature.tonnetz(y = y, sr = sr)
    
    songdata = np.array([np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])
    for e in mfcc:
        songdata = np.append(songdata, np.mean(e))
    for f in chroma_cqt :
        songdata = np.append(songdata, np.mean(f))
    for h in chroma_cens :
        songdata = np.append(songdata, np.mean(h))
    for i in spec_con :
        songdata = np.append(songdata, np.mean(i))
    for j in spec_flat :
        songdata = np.append(songdata, np.mean(j))
    for k in tonnetz :
        songdata = np.append(songdata, np.mean(k))  
     
    songdata = songdata.reshape(1, -1)
    songdata = scaler.transform(songdata)
    songpred = model.predict(songdata)
    
    if encoder is  None :
        print(songname, " was identified as : " , songpred[0])
    
    else :
        songpred = encoder.transform(songpred)
        print(songname, " was identified as : " , songpred[0])
        
        
        
        
        
def predict_genre_knn(model, scaler, songname, encoder = None):
    
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for i in range(1, 13):
        header += f' cqt{i}'
    for i in range(1, 13):
        header += f' cens{i}'
    for i in range(1, 8):
        header += f' cont{i}'
    header += ' spectram_flatness'
    for i in range(1, 7):
        header += f' tonnetz{i}'
    header += ' label'
    header = header.split()
    
    
    
    file = open('dataforpred.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    y, sr = librosa.load(songname, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    #added
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr = sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr = sr)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(y=y)
    tonnetz = librosa.feature.tonnetz(y = y, sr = sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    for f in chroma_cqt :
        to_append += f' {np.mean(f)}'
    for h in chroma_cens :
        to_append += f' {np.mean(h)}'
    for i in spec_con :
        to_append += f' {np.mean(i)}'
    for j in spec_flat :
        to_append += f' {np.mean(j)}'
    for k in tonnetz :
        to_append += f' {np.mean(k)}'
    to_append += f' {g}'
    file = open('dataforpred.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
            
            
    dataset_file = "dataforpred.csv"

    songdata = pd.read_csv(dataset_file, sep=",")         
   
    songdata = songdata.drop(["label", "filename",'mfcc15', 'mfcc19', 'mfcc18', 'mfcc20', 'mfcc6', 'cont3',
       'mfcc7', 'mfcc8', 'cqt1', 'cqt6', 'mfcc10', 'mfcc2',
       'zero_crossing_rate', 'cqt5', 'mfcc11', 'cens12', 'spectral_centroid',
       'cens2', 'mfcc13', 'cont7', 'tonnetz2', 'cqt12', 'tonnetz6', 'cqt10',
       'cqt9', 'tonnetz5', 'cens8', 'cens5', 'cqt4', 'cens4', 'cens11', 'cqt3',
       'tonnetz4', 'mfcc16', 'cens3', 'cqt8', 'tonnetz3', 'cens7', 'cqt7',
       'cqt2', 'cens1', 'cens9', 'cens10', 'cens6'], axis=1).values
    
    
    songdata = scaler.transform(songdata)
    songpred = model.predict(songdata)
    
    
    if encoder is  None :
        print(songname, " was identified as : " , songpred[0])
    
    else :
        songpred = encoder.transform(songpred)
        print(songname, " was identified as : " , songpred[0]) 
        
        
        
def predict_genre_lsvc(model, scaler, songname, encoder = None):
    
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for i in range(1, 13):
        header += f' cqt{i}'
    for i in range(1, 13):
        header += f' cens{i}'
    for i in range(1, 8):
        header += f' cont{i}'
    header += ' spectram_flatness'
    for i in range(1, 7):
        header += f' tonnetz{i}'
    header += ' label'
    header = header.split()
    
    
    
    file = open('dataforpredlsvc.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    y, sr = librosa.load(songname, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    #added
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr = sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr = sr)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    spec_flat = librosa.feature.spectral_flatness(y=y)
    tonnetz = librosa.feature.tonnetz(y = y, sr = sr)
    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    for f in chroma_cqt :
        to_append += f' {np.mean(f)}'
    for h in chroma_cens :
        to_append += f' {np.mean(h)}'
    for i in spec_con :
        to_append += f' {np.mean(i)}'
    for j in spec_flat :
        to_append += f' {np.mean(j)}'
    for k in tonnetz :
        to_append += f' {np.mean(k)}'
    to_append += f' {g}'
    file = open('dataforpredlsvc.csv', 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
            
            
    dataset_file = "dataforpredlsvc.csv"

    songdata = pd.read_csv(dataset_file, sep=",")         
   
    songdata = songdata.drop(["label", "filename",'cqt9', 'cens11', 'cens12', 'tonnetz4', 'mfcc13', 'cens2', 'mfcc20',
       'cqt2', 'cens4', 'mfcc10', 'tonnetz2', 'cqt12', 'tonnetz1', 'cens9',
       'cens7', 'cqt7', 'cens5', 'tonnetz3', 'cens6', 'cqt3', 'tonnetz6',
       'cens3', 'cens8', 'cqt10', 'tonnetz5', 'cens10', 'cens1'], axis=1).values
    
    
    songdata = scaler.transform(songdata)
    songpred = model.predict(songdata)
    
    
    if encoder is  None :
        print(songname, " was identified as : " , songpred[0])
    
    else :
        songpred = encoder.transform(songpred)
        print(songname, " was identified as : " , songpred[0])        