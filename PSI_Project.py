
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import genfromtxt
from sklearn.linear_model import LogisticRegression
numpy.set_printoptions(threshold=numpy.inf)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# load data
url = "forestfiresv3.csv"
names = ['X', 'Y',
        'month', 'day',
        'FFMC', 'DMC',
        'DC', 'ISI',
        'temp', 'RH',
        'wind', 'rain', 'area', 'cos tam']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values

X = array[:, 0:12]
X=X.astype('float')
Y =  array[:, 12]
Y = Y.astype('float')

# feature extraction
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, Y)

selected_columns = test.get_support(indices=True)
# summarize scores
numpy.set_printoptions(precision=3)

features = fit.transform(X)
# summarize selected feature
print(features[0:5,:])

df = pandas.DataFrame(data=features)
df.to_csv('predictedforest5.csv', index=False)