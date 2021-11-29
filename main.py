import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category = FutureWarning)


data = pd.read_csv('collegePlace.csv')

data.head()
print(f"Shape of Dataframe is: {data.shape}")
print('Datatype in Each Column\n')
pd.DataFrame(data.dtypes, columns=['Datatype']).rename_axis("Column Name")
data.describe().T.style.bar(subset=['mean'], color='#205ff2').background_gradient(subset=['std'], cmap='Reds').background_gradient(subset=['50%'], cmap='coolwarm')
pd.DataFrame(data.isnull().sum(), columns=["Null Values"]).rename_axis("Column Name")
fig = px.histogram(data, 'Age',
                   title="<b>Average Age of Student</b>")

fig.add_vline(x=data['Age'].mean(), line_width=2, line_dash="dash", line_color="red")

fig.show()
fig = px.histogram(data, 'Age',             
                   color="Gender",
                   title="<b>Average Age Gender wise</b>")

fig.add_vline(x=data['Age'].mean(), line_width=2, line_dash="dash", line_color="black")

fig.show()
pd.DataFrame(data['Gender'].value_counts()).rename({"Gender":"Counts"}, axis = 1).rename_axis("Gender")
px.histogram(data, x = "Gender", title = "<b>Total Male and Female</b>")
fig = px.pie(data, names = "Gender",
             title = "<b>Counts in Gender</b>",
             hole = 0.5, template = "plotly_dark")

fig.update_traces(textposition='inside',
                  textinfo='percent+label',
                  marker=dict(line=dict(color='#000000', width = 1.5)))


fig.show()
male = data[data['Gender'] == "Male"]
female = data[data['Gender'] == "Female"]
total_male = male.shape[0]
total_female = female.shape[0]
total_male_pass = male[male['PlacedOrNot'] == 1].shape[0]
total_female_pass = female[female['PlacedOrNot'] == 1].shape[0]
pass_male_percentage = np.round((total_male_pass * 100) / total_male,2)
pass_female_percentage = np.round((total_female_pass * 100) / total_female,2)

details = {"Total Male": [total_male],
             "Total Female": [total_female],
             "Total male pass" : [total_male_pass],
             "Total female pass" : [total_female_pass],
             "% of Passed Male" : [pass_male_percentage],
             "% of Passed Female" : [pass_female_percentage]}
      
details
gender_wise = pd.DataFrame(details, index=["Detail"])
gender_wise.T

fig = px.histogram(data_frame = data,
             x = "Stream",
             color="PlacedOrNot", title="<b>Counts of Stream</b>",
             pattern_shape_sequence=['x'],
             template='plotly_dark')

fig.show()

cgpa_above_avg = data[data['CGPA'] > data['CGPA'].mean()]

cgpa_above_avg

fig = px.histogram(data_frame = cgpa_above_avg,
                   x = 'CGPA',
                   color='PlacedOrNot',
                   title = "<b>Above Average CGPA Vs Placement</b>",
                   template='plotly')

fig.update_layout(bargap=0.2)

fig.show()

stream_wise = data.groupby('Stream').agg({'Age':'mean',
                                          'Internships' : 'sum',                            
                                           "CGPA":'mean',
                                           'PlacedOrNot':'sum'})

stream_wise.style.highlight_max()

px.bar(data_frame=stream_wise, barmode='group',
       title = "<b>Stream wise Analyzing</b>",template="plotly_dark")
       
no_internship = data[data['Internships'] == 0]

no_internship

fig = px.histogram(data_frame = no_internship,
                   x = "PlacedOrNot",
                   color="PlacedOrNot",
                   title = "<b>No Internship Experience Vs Placement</b>")

fig.update_layout(bargap=0.2)

fig.show()

fig = px.pie(no_internship, names = "PlacedOrNot",
             hole = 0.5)

fig.update_traces(textposition='inside',
                  textinfo='percent+label',
                  marker=dict(line=dict(color='#000000', width = 1.5)))


fig.show()

dummy_gender = pd.get_dummies(data['Gender'])
dummy_stream = pd.get_dummies(data['Stream'])

data = pd.concat([data.drop(["Gender", "Stream"], axis = 1), dummy_gender, dummy_stream], axis = 1)

data

data = data[['Age', 'Male', 'Female',
             'Electronics And Communication',
             'Computer Science', 'Information Technology',
             'Mechanical', 'Electrical', "Civil",
             "Internships","CGPA",'Hostel',
             'HistoryOfBacklogs', 'PlacedOrNot']]

data
scaler = StandardScaler()

scaler.fit(data.drop('PlacedOrNot',axis=1))

scaled_features = scaler.transform(data.drop('PlacedOrNot',axis=1))

scaled_features = pd.DataFrame(scaled_features, columns = data.columns[:-1])
scaled_features.head()
corrmat = data.corr()
top_corr_features = corrmat.index

plt.figure(figsize=(20,15))

#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")corrmat = data.corr()
top_corr_features = corrmat.index

plt.figure(figsize=(20,15))

#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features,
                                                    data['PlacedOrNot'],
                                                    test_size = 0.25,
                                                    random_state = 0)
                                                    
def models_score(models, X_train, X_test, y_train, y_test):    
    
    scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores[name] = model.score(X_test,y_test)

    model_scores = pd.DataFrame(scores, index=['Score']).transpose()
    model_scores = model_scores.sort_values('Score')
        
    return model_scores
    
models = {"DecisionTree":DecisionTreeClassifier(),
         "RandomForest":RandomForestClassifier(),
         "KNeighborsClassifier":KNeighborsClassifier()}
         
 model_scores = models_score(models, X_train, X_test, y_train, y_test)
 model_scores.style.highlight_max()
 
 model_scores = model_scores.reset_index().rename({"index":"Algorithms"}, axis = 1)

model_scores.style.bar()

fig = px.bar(data_frame = model_scores,
             x="Algorithms",
             y="Score",
             color="Algorithms", title = "<b>Models Score</b>", template = 'plotly_dark')

fig.update_layout(bargap=0.2)

fig.show()

label = model_scores['Algorithms']
value = model_scores['Score']

fig = go.Figure(data=[go.Pie(labels = label, values = value, rotation = 90)])

fig.update_traces(textposition='inside',
                  textinfo='percent+label',
                  marker=dict(line=dict(color='#000000', width = 1.5)))

fig.update_layout(title_x=0.5,
                  title_font=dict(size=20),
                  uniformtext_minsize=15)

fig.show()

