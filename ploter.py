import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




#Takes a combined dataframe of all the results

#Gmean plot


colors = ["windows blue", "orange red", "light brown", "amber", 'purple', 'jade', 'grey']  
myPalette = sns.xkcd_palette(colors) #passing colors to xkcd_palette function

sns.set(style="white") #white background
g = sns.factorplot(x="samp_technique", y="g_mean", hue="classifier", data=combined_df1, saturation=5, size=5, aspect=2.4, kind="bar",
          palette= myPalette, legend=False,) #removes legend

g.set(ylim=(0, 1)) 
g.despine(right=False) 
g.set_xlabels("") 
g.set_ylabels("G-mean Score")  
g.set_yticklabels("") 


#Matplotlib --legend creation

myLegend=plt.legend(bbox_to_anchor=(0., 1.2, 1., .102), prop ={'size':7.5}, loc=10, ncol=3, #3 rows per legend 
                                        #left, bottom, width, height
        title=r'ROC Score per sampling technique and classifier')                    
myLegend.get_title().set_fontsize('24')




ax=g.ax
for p in ax.patches:
    ax.annotate("%.4f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='center', fontsize=11.5 , color='black', rotation=90, xytext=(0, 20),
        textcoords='offset points')

plt.show()
