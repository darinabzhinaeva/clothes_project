from lxml import html
import requests
import json

url = "https://www2.hm.com/ru_ru/muzhchiny/vybrat-kategoriyu/dzhinsy.html?product-type=men_jeans&sort=stock&image-size=small&image=model&offset=0&page-size=117"

# Add headers for request
headers = {"User-Agent": "Mozilla/5.0"}

# Send request
response = requests.get(url, headers=headers)

# Convert response to text
html_text = str(response.text)
# Create lxml.html.HtmlElement object wich contains our HTML in structered view.
tree = html.fromstring(html_text)
# Select all our articles by using css selector
#assert isinstance(tree.cssselect, object)
articles = tree.cssselect("li.product-item article.hm-product-item")

url_list = []

for article in articles:
    # Get page url
    pages = article.cssselect(".list-swatches a")
    for page in pages:
        data = page.get("href")
        if data is not None:
            url_list.append(data)

print(url_list)
url_list = set(url_list)
url_list = list(url_list)

count = 0
data_list = []
for url in url_list:

    data = {}
    data["url"] = url

    # Send request
    response = requests.get("https://www2.hm.com/" + url, headers=headers)

    # Convert response to text
    html_text = str(response.text)

    # Create lxml.html.HtmlElement object wich contains our HTML in structered view.
    tree = html.fromstring(html_text)

    images = tree.cssselect("img")

    hrefs = []
    if len(images) > 0:
        for image in images:
            href = image.get("src")
            if href:
                hrefs.append(href)

    for i, line in enumerate(hrefs):
        path = 'hm/image_' + str(count)
        count +=1
        try:
            link = "https:" + line
            img = requests.get(link)
        except:
            continue
        # if we get status code 200, it means "all fine" and image was download normally
        if img.status_code == 200:
            with open(path + ".jpg", "wb") as imgfile:
                imgfile.write(img.content)
