import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

train_data = pd.read_csv('./Datasets/train_set.csv')
test_data = pd.read_csv('./Datasets/test_set.csv')

ftrLE = LabelEncoder()
train_data.FTR = ftrLE.fit_transform(train_data.FTR)
test_data.FTR = ftrLE.transform(test_data.FTR)

ftrOHE = OneHotEncoder()
ftrOHE.fit(train_data.FTR.values.reshape(-1, 1))

train_OHE = ftrOHE.transform([[each] for each in train_data.FTR]).toarray()
test_OHE = ftrOHE.transform([[each] for each in test_data.FTR]).toarray()

formLE = LabelEncoder()
col_to_transform = ['HM5', 'AM5']

possible_labels = ['W', 'D', 'L', 'M']
formLE.fit(possible_labels)


def encode_characters(string):
    return [formLE.transform([character])[0] if character in possible_labels else -1 for character in string]


# Encode characters in HM5 and AM5 and replace original columns
for i in range(1, 6):
    train_data[f'HM{i}'] = train_data['HM5'].apply(lambda x: encode_characters(x)[i-1])
    test_data[f'HM{i}'] = test_data['HM5'].apply(lambda x: encode_characters(x)[i-1])
    train_data[f'AM{i}'] = train_data['AM5'].apply(lambda x: encode_characters(x)[i-1])
    test_data[f'AM{i}'] = test_data['AM5'].apply(lambda x: encode_characters(x)[i-1])


def plot_correlation(data):
    plt.figure(figsize=(15, 10))
    plt.style.use('ggplot')
    my_cmap = plt.get_cmap('Accent')

    selected_features = ['FTR', 'HTGS_norm', 'ATGS_norm', 'HTGC_norm', 'ATGC_norm',
                         'HTP_norm', 'ATP_norm', 'HM1', 'AM1', 'HM2', 'AM2',
                         'HM3', 'AM3', 'HM4', 'AM4', 'HM5', 'AM5', 'HomeTeamRk',
                         'AwayTeamRk', 'DiffLP']

    numeric_df = data[selected_features]

    # Calculate correlations excluding 'FTR' column
    correlations = numeric_df.drop('FTR', axis=1).corr()['FTR']

    # Remove NaN values and sort correlations
    correlations = correlations.dropna().sort_values(ascending=False)

    # Plot correlations
    correlations.plot(kind='bar', cmap=my_cmap)

    plt.title('Correlation', fontsize=20)
    plt.xlabel('Features', fontsize=20)
    plt.ylabel('Relevance', fontsize=20)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.savefig('./correlation_plot.png', format='png')
    plt.show()

    print('Most Positive Correlations: \n', correlations.head(5))
    print('\nMost Negative Correlations: \n', correlations.tail(5))


def plot_correlation_heatmap(data):
    selected_features = ['HTGS_norm', 'ATGS_norm', 'HTGC_norm', 'ATGC_norm',
                         'HTP_norm', 'ATP_norm', 'HM1', 'AM1', 'HM2', 'AM2',
                         'HM3', 'AM3', 'HM4', 'AM4', 'HM5', 'AM5', 'HomeTeamRk',
                         'AwayTeamRk', 'DiffLP']

    numeric_df = data[selected_features].select_dtypes(include=[np.number])
    cor = numeric_df.corr()

    plt.figure(figsize=(40, 35))
    ax = sns.heatmap(cor, annot=False, cmap='coolwarm')

    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=45, rotation=45)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=45, rotation=0)

    ax.figure.axes[-1].tick_params(labelsize=45)

    plt.savefig('./correlation_matrix_selected.png', format='png')

    plt.show()


# 100: Away, 001: Home, 010: Draw
train_data['FTR_OHE1'] = train_OHE[:, 0]
train_data['FTR_OHE2'] = train_OHE[:, 1]
train_data['FTR_OHE3'] = train_OHE[:, 2]

test_data['FTR_OHE1'] = test_OHE[:, 0]
test_data['FTR_OHE2'] = test_OHE[:, 1]
test_data['FTR_OHE3'] = test_OHE[:, 2]

train_data.to_csv('./Datasets/train_set_en.csv', index=False)
test_data.to_csv('./Datasets/test_set_en.csv', index=False)
# pd.DataFrame(train_OHE).to_csv('./Datasets/train_onehot.csv', index=False)
# pd.DataFrame(test_OHE).to_csv('./Datasets/test_onehot.csv', index=False)
# plot_correlation(train_data)
# plot_correlation_heatmap(train_data)
