#!/usr/bin/env python
# coding: utf-8

# In[8]:


# we need that the price of 2 kilos of apple and 3 of pear and 4 of limes is 12.25 pounds
# so we assume the price of these fruits according to this price

fruitprice = {'apples': 1.00 ,'pears': 1.75 , 'limes': 1.25 , 'bananas':2.00 , 'oranges': 0.75 }


# In[22]:


def buylotsofFruit(orderlist):
    totalprice=0.00
    totalprice = ( 1*orderlist[0][1] +1.75*orderlist[1][1] +1.25*orderlist[2][1] )
    return totalprice


# In[23]:


testlist = [('apples',2.0),('pears',3.0),('limes',4.0)]
try:
    print("the cost of ",testlist , "is ", buylotsofFruit(testlist))
except:
    print("error : the fruit you entered is not in the list of prices !")

