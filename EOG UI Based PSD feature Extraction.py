import os
import pickle

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.signal import butter, filtfilt, resample, periodogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier


def butter_bandpass_filter(input_signal, lowcf, highcf, sr, order):
    nyq = 0.5*sr
    low = lowcf/nyq
    high = highcf/nyq
    numerator, denominator = butter(order, [low, high], btype='band', output='ba', analog=False, fs=None)
    filtered = filtfilt(numerator, denominator, input_signal)
    return filtered

# preparaing Data.


names = os.listdir("3-class")
up_h = []
up_v = []
down_h = []
down_v = []
left_h = []
left_v = []
right_h = []
right_v = []
blink_h = []
blink_v = []

for path in names:
    file = open("3-class/"+path)
    if path.__contains__('asagi'):
        lines = file.readlines()
        match = re.search(r'\d+', path)
        number = int(match.group())
        amp = [number]
        for i in range(len(lines) - 1):
            L = lines[i]
            amp.append(int(L))
        if path.__contains__('h'):
            down_h.append(amp)
        else:
            down_v.append(amp)
    if path.__contains__('sag') and path.startswith('sa'):
        lines = file.readlines()
        match = re.search(r'\d+', path)
        number = int(match.group())
        amp = [number]
        for i in range(len(lines) - 1):
            L = lines[i]
            amp.append(int(L))
        if path.__contains__('h'):
            right_h.append(amp)
        else:
            right_v.append(amp)
    if path.__contains__('sol'):
        lines = file.readlines()
        match = re.search(r'\d+', path)
        number = int(match.group())
        amp = [number]
        for i in range(len(lines) - 1):
            L = lines[i]
            amp.append(int(L))
        if path.__contains__('h'):
            left_h.append(amp)
        else:
            left_v.append(amp)
    if path.__contains__('kirp'):
        lines = file.readlines()
        match = re.search(r'\d+', path)
        number = int(match.group())
        amp = [number]
        for i in range(len(lines) - 1):
            L = lines[i]
            amp.append(int(L))
        if path.__contains__('h'):
            blink_h.append(amp)
        else:
            blink_v.append(amp)
    if path.__contains__('yukari'):
        lines = file.readlines()
        match = re.search(r'\d+', path)
        if match:
            number = int(match.group())
        else:
            number = 0
        amp = [number]
        for i in range(len(lines) - 1):
            L = lines[i]
            amp.append(int(L))
        if path.__contains__('h'):
            up_h.append(amp)
        else:
            up_v.append(amp)

up_h = sorted(up_h, key=lambda x: x[0])
up_v = sorted(up_v, key=lambda x: x[0])
down_h = sorted(down_h, key=lambda x: x[0])
down_v = sorted(down_v, key=lambda x: x[0])
left_h = sorted(left_h, key=lambda x: x[0])
left_v = sorted(left_v, key=lambda x: x[0])
right_h = sorted(right_h, key=lambda x: x[0])
right_v = sorted(right_v, key=lambda x: x[0])
blink_h = sorted(blink_h, key=lambda x: x[0])
blink_v = sorted(blink_v, key=lambda x: x[0])

# UP H & V
modified = []
for i in up_h:
    signal = i[1:]
    modified.append(signal)
up_h = modified

modified = []
for i in up_v:
    signal = i[1:]
    modified.append(signal)
up_v = modified

# Down H & V
modified = []
for i in down_h:
    signal = i[1:]
    modified.append(signal)
down_h = modified

modified = []
for i in down_v:
    signal = i[1:]
    modified.append(signal)
down_v = modified

# left H & V
modified = []
for i in left_h:
    signal = i[1:]
    modified.append(signal)
left_h = modified

modified = []
for i in left_v:
    signal = i[1:]
    modified.append(signal)
left_v = modified

# Right H & V
modified = []
for i in right_h:
    signal = i[1:]
    modified.append(signal)
right_h = modified

modified = []
for i in right_v:
    signal = i[1:]
    modified.append(signal)
right_v = modified

# Blink H & V
modified = []
for i in blink_h:
    signal = i[1:]
    modified.append(signal)
blink_h = modified

modified = []
for i in blink_v:
    signal = i[1:]
    modified.append(signal)
blink_v = modified

up = [HEOG + VEOG for HEOG, VEOG in zip(up_h, up_v)]
down = [HEOG + VEOG for HEOG, VEOG in zip(down_h, down_v)]
left = [HEOG + VEOG for HEOG, VEOG in zip(left_h, left_v)]
right = [HEOG + VEOG for HEOG, VEOG in zip(right_h, right_v)]
blink = [HEOG + VEOG for HEOG, VEOG in zip(blink_h, blink_v)]

samples = []
targets = []
for signal in up:
    samples.append(signal)
    targets.append('up')

for signal in down:
    samples.append(signal)
    targets.append('down')

for signal in left:
    samples.append(signal)
    targets.append('left')

for signal in right:
    samples.append(signal)
    targets.append('right')

for signal in blink:
    samples.append(signal)
    targets.append('blink')

# Preprocessing
# filter for signals
filtered_samples = []
for signal in samples:
    filtered_sample = butter_bandpass_filter(signal, lowcf=1, highcf=30, sr=176, order=2)
    filtered_samples.append(filtered_sample)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(np.arange(0, 500), samples[0])
plt.title('before filtering for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')

plt.subplot(122)
plt.plot(np.arange(0, 500), filtered_samples[0])
plt.title('After filtering for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')
plt.show()

# Resampling >2Freq_MAx(60)
resampled_signals = []
for signal in filtered_samples:
    resampledsignal = resample(signal, 80)
    resampled_signals.append(resampledsignal)

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(np.arange(0, 500), filtered_samples[0])
plt.title('Before DownSampling for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')

plt.subplot(122)
plt.plot(np.arange(0, 80), resampled_signals[0])
plt.title('After DownSampling for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')
plt.show()

# Normalization
normalized_signals = []
for signal in resampled_signals:
    normlizedsignal = [(x-min(signal))/(max(signal)-min(signal)) for x in signal]
    normalized_signals.append(normlizedsignal)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(np.arange(0, 80), resampled_signals[0])
plt.title('Before Normalizing for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')

plt.subplot(122)
plt.plot(np.arange(0, 80), normalized_signals[0])
plt.title('After Normalizing for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')
plt.show()

# remove DC Component
removed_DC_signals = []
for signal in normalized_signals:
    mean = statistics.mean(signal)
    removedDCsignal = [(x - mean) for x in signal]
    removed_DC_signals.append(removedDCsignal)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(np.arange(0, 80), normalized_signals[0])
plt.title('Before DC_Removal for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')

plt.subplot(122)
plt.plot(np.arange(0, 80), removed_DC_signals[0])
plt.title('After DC_Removal for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')
plt.show()

# Feature Extraction
features_of_signals = []
for signal in removed_DC_signals:
    (f, s) = periodogram(signal, 176, scaling='density')
    features_of_signals.append(s)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(np.arange(0, 80), removed_DC_signals[0])
plt.title('Before Feature Extraction for signal 0')
plt.xlabel('Time')
plt.ylabel('Amp')

plt.subplot(122)
plt.semilogy(np.arange(0, 41), features_of_signals[0])
plt.xlim(0, 50)
plt.ylim([1e-10, 1e2])
plt.title('After Feature Extraction for signal 0')
plt.xlabel('freq')
plt.ylabel('PSD')
plt.show()

numbers = []
for i in range(100):
    numbers.append(i)

df1 = pd.DataFrame(features_of_signals)
df2 = pd.DataFrame(targets)
df2.rename(columns={0: 'targets'}, inplace=True)
df3 = pd.DataFrame(numbers)
df3.rename(columns={0:'numbers'},inplace=True)
Data1 = pd.concat([df1, df2], axis=1)
Data = pd.concat([Data1,df3],axis=1)
Data = Data.sample(frac=1).reset_index(drop=True)
X = Data.drop({'targets','numbers'}, axis=1)
Y = Data['targets']


X_train, X_Val_test, Y_train, Y_Val_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

X_val, X_test, Y_val, Y_test = train_test_split(X_Val_test, Y_Val_test, test_size=0.5, random_state=42, shuffle=False)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
logreg_pred = logreg.predict(X_val)
logreg_acc = accuracy_score(logreg_pred, Y_val)
print('Test accuracy of Logistic Regression:', logreg_acc*100)
filename = 'Logistic Regression'
pickle.dump(logreg, open(filename, 'wb'))

# Linear SVM model
SVC = LinearSVC()
SVC.fit(X_train, Y_train)
SVC_pred = SVC.predict(X_val)
SVC_acc = accuracy_score(SVC_pred, Y_val)
print('Test accuracy SVM:', SVC_acc*100)
filename = 'SVM'
pickle.dump(SVC, open(filename, 'wb'))

# Decision Tree Model
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_val)
dt_acc = accuracy_score(dt_pred, Y_val)
print('Test accuracy of Decision Tree: ', dt_acc * 100)
filename = 'Decision Tree'
pickle.dump(dt, open(filename, 'wb'))

# Random Forest Model
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_val)
rf_acc = accuracy_score(rf_pred, Y_val)
print('Test accuracy of Random Forest: ', rf_acc * 100)
filename = 'Random Forest'
pickle.dump(rf, open(filename, 'wb'))

# Gradient Boosting Classifier Model
gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)
gb_pred = gb.predict(X_val)
gb_acc = accuracy_score(gb_pred, Y_val)
print('Test accuracy of Gradient Boosting Classifier: ', gb_acc * 100)
filename = 'Gradient Boosting Classifier'
pickle.dump(gb, open(filename, 'wb'))

# Gaussian Model
Gaussian = GaussianNB()
Gaussian.fit(X_train, Y_train)
Gaussian_pred = Gaussian.predict(X_val)
Gaussian_acc = accuracy_score(Gaussian_pred, Y_val)
print('Test accuracy Of Gaussian: ', Gaussian_acc * 100)
filename = 'Gaussian'
pickle.dump(Gaussian, open(filename, 'wb'))

print('Test Data \n',Data.tail(10))