from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb
import numpy as np
import librosa


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
    songdata = np.array([np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])
    for e in mfcc:
        songdata = np.append(songdata, np.mean(e))
        
     
    songdata = songdata.reshape(1, -1)
    songdata = scaler.transform(songdata)
    songpred = model.predict(songdata)
    
    if encoder is  None :
        print(songname, " was identified as : " , songpred[0])
    
    else :
        songpred = encoder.transform(songpred)
        print(songname, " was identified as : " , songpred[0])