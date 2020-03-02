import pandas as pd
import urllib
csv = pd.read_csv("https://docs.google.com/spreadsheets/d/10T3lIOc5fZgsyvmYeWOlmgsaRiGI2BhYNBPY5YH0EFk/export?format=csv&id=10T3lIOc5fZgsyvmYeWOlmgsaRiGI2BhYNBPY5YH0EFk&gid=0")
csv.dropna()
csv.drop_duplicates()
for ind in csv.index:
    try:
        print(str(csv["titles"][ind]))

        urllib.request.urlretrieve("https://i.ytimg.com/vi/"+str(csv["id"][ind])+"/hqdefault.jpg", "c:/Users/Skull/Documents/github/Thumbnail-clickbait-detection/data_manualy_tag/"+str(csv["total"][ind])+"_"+str(csv["id"][ind])+".jpg")
    except Exception as e:
        print("error on :"+str(csv["titles"][ind]))
