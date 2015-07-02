df=..
df['Occupancy']=0
Cabin_list=df.Cabin.dropna().unique()
for Cabin in Cabin_list:
	nb=len(df[df.Cabin==Cabin])
	df.loc[(Cabin==df.Cabin),'Occupancy']=nb

