import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
df=pickle.load(open('dfIPL.pkl' , 'rb'))
pipe=pickle.load(open('pipe.pkl','rb'))

df=df[df["balls_left"]!=0]
df.dropna(inplace=True)
x=df.drop("winner" , axis=1)
y=df["winner"]

x_train , x_test , y_train , y_test =train_test_split(x,y , test_size=0.2)

step1 = ColumnTransformer(transformers=[
    ("col_tnf", OneHotEncoder(sparse=False, drop="first", handle_unknown="ignore"), [0, 1, 8])

]
    , remainder="passthrough")
step2=LogisticRegression(solver="liblinear")
pipe=Pipeline([
    ("step1",step1),
    ("step2",step2)
])
pipe.fit(x_train , y_train )

#pre=pickle.load(open('PredictorIPL.pkl' , 'rb'))
st.title("IPL Win Probablity")

col1 , col2 =st.columns(2)

with col1:
    batting_team=st.selectbox('Batting Team' , df["batting_team"].unique())
with col2:
    bowling_team=st.selectbox("Bowling Team" , df["bowling_team"].unique())

city=st.selectbox("City" , df["city"].unique())

Target=st.number_input("Target")

col3 , col4 , col5=st.columns(3)

with col3:
    score=st.number_input("Score")
with col4:
    Over=st.number_input("Over_Completed")
with col5:
    wicket=st.number_input("Wicket")


if st.button("Show Winning Probablity"):
    runs_left=Target-score
    balls_left=120-(Over*6)
    wicket=10-wicket
    crr=score/Over
    rrr=(runs_left*6)/balls_left

    input_df=pd.DataFrame({"batting_team":[batting_team] ,"bowling_team":[bowling_team] ,"left_runs":[runs_left]
                   , "balls_left":[balls_left] , "total_runs_y":[Target],"Wicketleft":[wicket] ,"crr":[crr] , "rrr":[crr], "city":[city]})
    re=pipe.predict_proba(input_df)
    loss=re[0][0]
    win=re[0][1]
    st.header(batting_team + "-"+str(round(win*100)) + "%")
    st.header(bowling_team + "-"+str(round(loss*100))+"%")
