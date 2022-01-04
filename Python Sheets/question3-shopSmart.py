#!/usr/bin/env python
# coding: utf-8

# In[20]:


def shopsmart(diction1,diction2,orderlist):
    list1 = [(k, v) for k, v in diction1.items()]
    list2 = [(m, n) for m, n in diction2.items()]
    cost1=0.0
    cost2=0.0
    for i in range(len(orderlist)):
        cost1 = cost1 + list1[i][1]*orderlist[i][1]
        cost2 = cost2 + list2[i][1]*orderlist[i][1]
    if cost1 > cost2 :
        print("for orders ", orderlist ," the best shop is shop 2 ")
    else:
        print("for orders ", orderlist ," the best shop is shop 1 ")
    
    


# In[21]:


orders1 = [('apples', 1.0), ('oranges', 3.0)]
orders2 = [('apples', 3.0)]
dir1 = {'apples': 2.0, 'oranges': 1.0}
dir2 = {'apples': 1.0, 'oranges': 5.0}

query1 = shopsmart(dir1,dir2,orders1)


# In[22]:


query2 = shopsmart(dir1,dir2,orders2)


# In[ ]:




