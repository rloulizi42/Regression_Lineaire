import requests
from bs4 import BeautifulSoup

response = requests.get("https://fr.aliexpress.com/wholesale?catId=0&initiative_id=SB_20180415061602&SearchText=ecouteur")
soup = BeautifulSoup(response.text,'html.parser')
print(soup)

#ultag = soup.find('div', {"class" : "main-wrap gallery-mode"})

#print (ultag)

#response1 = requests.get('https:' + ultag.find('div').find('div', class_="img img-border").find('div').find('a')['href'])
#soup1 = BeautifulSoup(response1.text, 'html.parser')
#
#detailtag = soup1.find('div', id="j-detail-page").find('div', class_="detail-main-layout container util-clearfix").find('div', class_="col-main").find('div', class_="main-wrap").find('div', class_="main-content")
#.find('div', class_="ui-tab ui-tab-normal ui-switchable")
#.find('div', class_="ui-tab-body").find('div', class_="ui-tab-pane shipping-payment-main ui-switchable")
