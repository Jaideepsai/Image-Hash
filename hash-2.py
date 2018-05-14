
# coding: utf-8

# In[1]:


import sys
import hashlib
from PIL import Image
import requests
from io import BytesIO


# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

md5 = hashlib.md5()
sha1 = hashlib.sha1()



response = requests.get('https://a0.muscache.com/im/pictures/45880516/93bb5931_original.jpg?aki_policy=xx_large')
# img = Image.open(BytesIO(response.content))

with Image.open(BytesIO(response.content)) as f:
    while True:
        data = f.read(BUF_SIZE)
        if not data:
            break
        md5.update(data)
        sha1.update(data)

print("MD5: {0}".format(md5.hexdigest()))
print("SHA1: {0}".format(sha1.hexdigest()))

