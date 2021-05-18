import pandas as pd
import matplotlib.pyplot as plt
import numpy

df = pd.read_csv('data/counties.csv')
df1 = pd.read_csv('data/counties.csv')
df['date'] =pd.to_datetime(df['date'])  #data是读取的一个csv文件里面有列"下单时间"

#整体情况
groud1=df.groupby('date').sum()
print(groud1)
groud1.to_csv('analysisData/美国按时间整体情况.csv')
plt.plot(groud1)
plt.show()

#州的情况
groud2=df1.groupby(['date','state','county','cases','deaths']).filter(lambda x: any(x['date']=='2020/5/19'))
groud2=groud2.groupby('state').sum()
print(groud2)
groud2.to_csv('analysisData/美国各州的情况.csv')
plt.xticks(rotation=60)
plt.plot(groud2)
plt.show()

#县的情况
groud4=df1.groupby(['date','state','county','cases','deaths']).filter(lambda x: any(x['date']=='2020/5/19'))
groud4=groud4.groupby(['state','county']).sum()
print(groud4)
groud4.to_csv('analysisData/县区情况.csv')
df = pd.read_csv("analysisData/县区情况.csv")
sorted_df = df.sort_values(by = 'cases', ascending = False)
sorted_df.to_csv("analysisData/县区情况.csv", index = False)

#New York-New York City数据、情况
grouped1=df1.groupby(['date','state','county','cases','deaths']).filter(lambda x: any(x['state']=='New York'))
grouped1=grouped1.groupby(['date','state','county','cases','deaths']).filter(lambda x: any(x['county']=='New York City'))
grouped1.to_csv('analysisData/New York-New York City.csv',index=False)
print(grouped1)
df2 = pd.read_csv('analysisData/New York-New York City.csv')
df2['date'] =pd.to_datetime(df2['date'])
groud3=df2.groupby('date').sum()
print(groud3)
plt.plot(groud3)
plt.show()

#Illinois Cook数据、情况
grouped2=df1.groupby(['date','state','county','cases','deaths']).filter(lambda x: any(x['state']=='Illinois'))
grouped2=grouped2.groupby(['date','state','county','cases','deaths']).filter(lambda x: any(x['county']=='Cook'))
grouped2.to_csv('analysisData/Illinois Cookk.csv',index=False)
print(grouped2)
df2 = pd.read_csv('analysisData/Illinois Cook')
df2['date'] =pd.to_datetime(df2['date'])
groud3=df2.groupby('date').sum()
print(groud3)
plt.plot(groud3)
plt.show()

#Virginia-Brunswick数据、情况
grouped3=df1.groupby(['date','state','county','cases','deaths']).filter(lambda x: any(x['state']=='Virginia'))
grouped3=grouped3.groupby(['date','state','county','cases','deaths']).filter(lambda x: any(x['county']=='Brunswick'))
grouped1.to_csv('analysisData/Virginia-Brunswick.csv',index=False)
print(grouped3)
df2 = pd.read_csv('analysisData/Virginia-Brunswick.csv')
df2['date'] =pd.to_datetime(df2['date'])
groud4=df2.groupby('date').sum()
print(groud4)
plt.plot(groud4)
plt.show()

