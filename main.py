import streamlit as st
import pandas as pd
import altair as alt
#import sklearn as sk
from sklearn.linear_model import LinearRegression, Lasso

# set web
st.set_page_config(
    page_title="BusinessCourseDataAnalysis",
    layout="wide",
    initial_sidebar_state="auto",
)
# read data files
df = pd.read_csv("BusinessFinance.csv")
df_initial = df.copy(deep=True)

# deal columns
df.columns = df.columns
df.drop(df.columns[[0, 1, 2, 3, 11, 12, 13, 14, 15]], axis=1, inplace=True)


# deal publishedTime
def setDateTime(datetime):
    times = datetime.split('T')
    times[1] = times[1][:-1]
    return times[0] + " " + times[1]


df['publishedTime'] = df.apply(lambda x: setDateTime(x['publishedTime']), axis=1)
df['publishedTime'] = pd.to_datetime(df['publishedTime'], format='%Y-%m-%d %H:%M:%S')
# only keep year
df['publishedTime'] = df.apply(lambda x: x['publishedTime'].year, axis=1)


# deal course time
def setContentInfo(datetime):
    times = datetime.split()
    if len(times) == 1:
        return int(times[0])
    result = eval(times[0])
    try:
        if times[1] == 'hours':
            result *= 60
    except:
        print(datetime)
    return int(result)


df['contentInfo'] = df.apply(lambda x: setContentInfo(x['contentInfo']), axis=1)
# deal price
df = df.replace('Free', '0')
df['price'].astype('int')


analysis = st.sidebar.selectbox('Option', ['Data preprocessing', 'Visualization', 'Sklearn Predict','References and Link'])
st.write("# " + analysis)

if analysis == 'Data preprocessing':

    num = st.slider('num', 0, len(df_initial))
    st.write(f"You have selected {num} inital data to show")
    st.write(df_initial.head(num))
    num_2 = st.slider('num_2', 0, len(df))
    st.write(f"You have selected {num_2} Preprocessed data to show")
    st.write(df.head(num_2))

elif analysis == 'Visualization':
    st.write("## The first topic we want to explore is:Which year has the most courses published?")
    df_year = df.copy(deep=True)
    df_year['publishedTime'] = df_year['publishedTime'].astype("str")
    st.write(df.head())
    chart_1 = alt.Chart(df_year).mark_bar().encode(
        x="publishedTime",
        y="count()",
        color=alt.Color('count()', scale=alt.Scale(scheme='turbo', reverse=True)),
    )
    st.altair_chart(chart_1, use_container_width=True)

    st.write("From the above chart, we can clearly know that the most courses published in 2016.\n")

    st.write("## The second topic we want to explore is:With the passage of time, what is the law of the change in "
             "the number of courses released for all levels?")

    chart_2 = alt.Chart(df).mark_line(point=True).encode(
        x="publishedTime",
        y="count()",
        color="instructionalLevel",
    )
    st.altair_chart(chart_2, use_container_width=True)
    st.write("Through the above chart, we find that the number of courses posted for all levels is increased and then "
             "decreased. The difference is that the time nodes for the decline occur first, followed by the first "
             "decline in All Level courses, and the latest decline in Expert Level courses.")

    chart_3 = alt.Chart(df).mark_circle().encode(
        x="numSubscribers",
        y="contentInfo",
        color=alt.Color('instructionalLevel', scale=alt.Scale(scheme='turbo')),
        tooltip=["price", "contentInfo"],
    )
    st.altair_chart(chart_3, use_container_width=True)

elif analysis == 'Sklearn Predict':
    # ML
    st.write(f"Let's look at ML how to deal this data")
    # value instructionalLevel
    level = {"Intermediate Level": 2, "Beginner Level": 1, "Expert Level": 3, "All Levels": 0}
    df['instructionalLevel'] = df.apply(lambda x: level[x['instructionalLevel']], axis=1)

    st.write(df.head())
    st.write("## The third topic we want to explore is:How to predict how many people will subscribe to a class after "
             "it is published in the future?")

    X = df[['numReviews', 'publishedTime', 'numPublishedLectures']]
    y = df['numSubscribers']
    st.write("value:")
    st.write(X.head())
    st.write("label:")
    st.write(y.head())

    LR = LinearRegression()
    L = Lasso(alpha=0.0005, random_state=0)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    LR.fit(X_train, y_train)
    L.fit(X_train, y_train)

    st.write("### Forecast effect evaluation")
    st.write("The accuracy of LinearRegression is: {:.5f}".format(LR.score(X_test, y_test)))
    st.write("The accuracy of Lasso is: {:.5f}".format(L.score(X_test, y_test)))
    pred_1 = pd.DataFrame(LR.predict(X), columns=['pred_LR'])
    pred_2 = pd.DataFrame(L.predict(X), columns=['pred_L'])
    y = pd.concat([y, pred_1, pred_2], axis=1)
    st.write("### The prediction result:")
    st.write(y.head())
    st.line_chart(y)
    st.write("By comparing the prediction results, it is found that the actual effect of pure linear model prediction "
             "is not ideal, and more detailed processing of the data set or the use of other more effective machine "
             "learning models are required.")
elif analysis == 'References and Link':
    st.write("GitHub: https://github.com/yuanchai9/BusinessCourseDataAnalysis")
    st.write("This portion of the app was taken from https://altair-viz.github.io/index.html")
    st.write("This portion of the app was taken from "
             "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html")
    st.write("This portion of the app was taken from "
             "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"
             "#sklearn.model_selection.train_test_split")
    st.write("This portion of the app was taken from "
        "https://docs.streamlit.io/library/api-reference/layout/st.sidebar")
    st.write("This portion of the app was taken from "
        "https://pandas.pydata.org/docs/reference/index.html")






