
# coding: utf-8

# In[1]:


# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()


# In[2]:


# a=[];b=[];c=[];d=[]
# for i in dirs:
#     if not i.startswith('.') and i != 'Thumbs.db':
#         a.append(i)
#         b.append(int(i.split('_')[1].split(".")[0]))
#         c.append(int(i.split('_')[0]))
#         d.append("old")
        
# df_old = pd.DataFrame(
#     {'file': a,
#      'image_number':b,
#      'file_split':c
#     })

# df_old.reset_index(drop=True)
# df_sort = df_old.groupby(['file_split'])

# res = df_sort.apply(lambda p: p.sort_values('image_number',ascending=True))

# res=res.reset_index(drop=True).drop_duplicates('file')
# res

# d=[];e=[];f=[];g=[]
# for j in dirs2:
#     if not j.startswith('.') and i != 'Thumbs.db':
#         d.append(j)
#         e.append(int(j.split('_')[1].split(".")[0]))
#         f.append(int(j.split('_')[0]))
#         g.append("new")
# df_new = pd.DataFrame(
#     {'file':d ,
#      'image_number':e,
#      'file_split':f
#     })

# df_new.reset_index(drop=True)
# df_sort_new = df_new.groupby(['file_split'])

# res_new = df_sort_new.apply(lambda x: x.sort_values('image_number',ascending=True))

# res_new=res_new.reset_index(drop=True).drop_duplicates('file')
# res_new


# In[3]:


path_old = "/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/old"
dirs = os.listdir( path_old )
path_new="/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/new"
dirs2=os.listdir( path_new )


a=[];b=[];c=[];d=[]
for i in dirs2:
    if not i.startswith('.') and i != 'Thumbs.db':
        if("verified" in i):
            a.append(i.split("_verified")[0]+"."+i.split(".")[1])
            b.append("yes")
            c.append(i.split("_verified")[0])
        else:
            a.append(i)
            b.append("No")
            c.append(i.split(".")[0])
        d.append("new_added")
                
df_new = pd.DataFrame(
    {'file': a,
     "verified":b,
     'image_type':d,
     'file_name':c
    })

d=[];e=[];f=[];g=[]
for j in dirs:
    if not j.startswith('.') and i != 'Thumbs.db':
        d.append(j)
        g.append("old_removed")
        e.append(j.split(".")[0])
df_old = pd.DataFrame(
    {'file':d ,
     'image_type':g,
     "verified":'nan',
     'file_name':e
    })

frames = [df_old,df_new]
conc=pd.concat(frames)

x=pd.concat(g for _, g in conc.groupby("file_name") if len(g) == 1)

for index,row in x.iterrows():
    if(row["verified"]=="yes"):   
         os.rename('/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/new'+'/'+row["file"].split(".")[0]+"_verified"+"."+row["file"].split(".")[1],'/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/'+row["image_type"]+'/'+row["file"].split(".")[0]+"_verified"+"."+row["file"].split(".")[1])
        
    elif(row["verified"]=="No"):
        
         os.rename('/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/new'+'/'+row["file"],'/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/'+row["image_type"]+'/'+row["file"])   
   
    else:
        os.rename('/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/old'+'/'+row["file"],'/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/'+row["image_type"]+'/'+row["file"])
           


# In[4]:


x.to_csv("changed_files.csv",sep=",")


# In[38]:


path_old = "/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/old"
dirs = os.listdir( path_old )
path_new="/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/new"
dirs2=os.listdir( path_new )
path_old_trim = "/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/old_removed"
dirs_trim = os.listdir( path_old_trim )
path_new_trim="/Users/jaideep/Desktop/Airbnb/Airbnb_Full_Project_files_2/Code/new_added"
dirs2_trim=os.listdir( path_new_trim )

y=pd.concat(q for _, q in conc.groupby("file_name") if len(q) != 1)
y=y.reset_index(drop=True)


# In[39]:


y


# In[58]:


if len(dirs)==len(dirs2):
    h=[];l=[];m=[];n=[]
    count=0
    for index,row in y.iterrows():
        if(row["image_type"]=="new_added"):
            if(row["verified"]=="yes"):
                image_path_1=path_old + "/" + row["file"]
                image_path_2=path_new + "/" + row["file"].split(".")[0]+"_verified."+row["file"].split(".")[1]
                original = cv2.imread(image_path_1)
                new = cv2.imread(image_path_2)
            else:
                image_path_3=path_old + "/" + row["file"]
                image_path_4=path_new + "/" + row["file"]
                original = cv2.imread(image_path_3)
                new = cv2.imread(image_path_4)
            original_trim = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            new_trim = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
            ssim_var=ssim(original_trim, new_trim)
#             mse=mse(original_trim, new_trim);
            l.append(row["file_name"])
            h.append(row["verified"])
            m.append(ssim_var)
#             n.append(mse)
            
df_compared = pd.DataFrame(
    {'file_name': l,
     'verified':h,
     "SSIM":m
    })


# In[59]:


df_compared.to_csv("image_comapred.csv",sep=",")

