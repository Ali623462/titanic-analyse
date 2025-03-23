import pandas as pd
import seaborn as sns
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_width', 1000)
df=sns.load_dataset("titanic")
df.head()
df.shape
df["sex"].value_counts()
df.nunique()
df["pclass"].unique()
df[["pclass", "parch"]].nunique()
df["embarked"].dtype
df["embarked"]=df["embarked"].astype("category")
df["embarked"].dtype
df.info()
df[df["embarked"]=="c"].head(10)
df[df["embarked"]!= "S"].head(10)
df[~(df["embarked"] == "S")]["embarked"].unique()
df[(df["age"]<30)&(df["sex"]=="female")].head()
df[(df["fare"]>500)|(df["age"]>70)].head()
df.isnull().sum()
df.drop("who", axis=1, inplace=True)
type(df["deck"].mode())
df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()
df["age"].fillna(df["age"].median(),inplace=True)
df.isnull().sum()
df.groupby(["pclass", "sex"]).agg({"survived":
["sum","count", "mean"]})
def age_30(age):
    if age<30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x : age_30(x))


df["age_flag"] = df["age"].apply(lambda x: 1 if x<30 else 0)
