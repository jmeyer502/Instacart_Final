
# Insta Cart Data


```python
# load basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import seaborn as sns
```


```python
del orders_df
del products_df
del orderp_df
del ordert_df
del aisles_df
del depts_df
```


```python
orders_df = pd.read_csv('orders.csv',  dtype={'order_id':'int32', 
                                                               'user_id':'int32',
                                                               'eval_set':'category', 
                                                               'order_dow':'int8', 
                                                               'order_hour_of_day':'int8', #could also be category
                                                               'days_since_prior_order':'float16'})
products_df = pd.read_csv('products.csv', dtype={'product_id':'int32', 
                                                                   'product_name':'category', 
                                                                   'aisle_id':'int16', 
                                                                   'department_id':'int16'})
ordert_df = pd.read_csv('order_products__train.csv', dtype={'order_id':'int32',
                                                                                     'product_id':'int32',
                                                                                     'add_to_cart_order':'int8',
                                                                                     'reordered':'uint8'})
orderp_df = pd.read_csv('order_products__prior.csv', dtype={'order_id':'int32',
                                                                                     'product_id':'int32',
                                                                                     'add_to_cart_order':'int8',
                                                                                     'reordered':'uint8'})
aisles_df = pd.read_csv('aisles.csv', 
                     dtype={'aisle_id':'int16', 'aisle':'category'})

depts_df = pd.read_csv('departments.csv',  
                          dtype={'department_id':'int8', 'department':'category'})
```

### Understand the datasets provided


```python
aisles_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aisle_id</th>
      <th>aisle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>prepared soups salads</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>specialty cheeses</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>energy granola bars</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>instant foods</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>marinades meat preparation</td>
    </tr>
  </tbody>
</table>
</div>




```python
aisles_df.shape
```




    (134, 2)




```python
products_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>aisle_id</th>
      <th>department_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Chocolate Sandwich Cookies</td>
      <td>61</td>
      <td>19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>All-Seasons Salt</td>
      <td>104</td>
      <td>13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Robust Golden Unsweetened Oolong Tea</td>
      <td>94</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Smart Ones Classic Favorites Mini Rigatoni Wit...</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Green Chile Anytime Sauce</td>
      <td>5</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
depts_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>department_id</th>
      <th>department</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>frozen</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>other</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>bakery</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>produce</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>alcohol</td>
    </tr>
  </tbody>
</table>
</div>




```python
depts_df.shape
```




    (21, 2)




```python
orderp_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>33120</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>28985</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9327</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>45918</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>30035</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
orderp_df.shape
```




    (32434489, 4)




```python
ordert_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
ordert_df.shape
```




    (1384617, 4)




```python
orders_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2398795</td>
      <td>1</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>473747</td>
      <td>1</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>12</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2254736</td>
      <td>1</td>
      <td>prior</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>431534</td>
      <td>1</td>
      <td>prior</td>
      <td>5</td>
      <td>4</td>
      <td>15</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
orders_df.shape
```




    (3421083, 7)




```python
g = orders_df.groupby(['eval_set'])
g[['order_id']].count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
    </tr>
    <tr>
      <th>eval_set</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>prior</th>
      <td>3214874</td>
    </tr>
    <tr>
      <th>test</th>
      <td>75000</td>
    </tr>
    <tr>
      <th>train</th>
      <td>131209</td>
    </tr>
  </tbody>
</table>
</div>



## Create Base Tables

#### Set up the order train file for use later on in the script


```python
ordert_df = ordert_df.merge(products_df, how = 'left', on = 'product_id') 
```


```python
ordert_df = ordert_df.merge(aisles_df, how = 'left', on = 'aisle_id')
```


```python
ordert_df = ordert_df.drop(['aisle_id','department_id'],axis=1) #axis one indicates columns versus rows
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-83-1efbcf37cc27> in <module>()
    ----> 1 ordert_df = ordert_df.drop(['aisle_id','department_id'],axis=1) #axis one indicates columns versus rows
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3692                                            index=index, columns=columns,
       3693                                            level=level, inplace=inplace,
    -> 3694                                            errors=errors)
       3695 
       3696     @rewrite_axis_style_signature('mapper', [('copy', True),
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3106         for axis, labels in axes.items():
       3107             if labels is not None:
    -> 3108                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       3109 
       3110         if inplace:
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\generic.py in _drop_axis(self, labels, axis, level, errors)
       3138                 new_axis = axis.drop(labels, level=level, errors=errors)
       3139             else:
    -> 3140                 new_axis = axis.drop(labels, errors=errors)
       3141             dropped = self.reindex(**{axis_name: new_axis})
       3142             try:
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexes\base.py in drop(self, labels, errors)
       4385             if errors != 'ignore':
       4386                 raise KeyError(
    -> 4387                     'labels %s not contained in axis' % labels[mask])
       4388             indexer = indexer[~mask]
       4389         return self.delete(indexer)
    

    KeyError: "labels ['aisle_id' 'department_id'] not contained in axis"



```python
ordert_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
      <td>Bulgarian Yogurt</td>
      <td>yogurt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
      <td>Organic 4% Milk Fat Whole Milk Cottage Cheese</td>
      <td>other creams cheeses</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Celery Hearts</td>
      <td>fresh vegetables</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
      <td>Cucumber Kirby</td>
      <td>fresh vegetables</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
      <td>Lightly Smoked Sardines in Olive Oil</td>
      <td>canned meat seafood</td>
    </tr>
  </tbody>
</table>
</div>




```python
# adding the user_id from the orders table
ordert_df = pd.merge(ordert_df, orders_df[['order_id','user_id']], how = 'left', on = 'order_id')
```


```python
ordert_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
      <td>Bulgarian Yogurt</td>
      <td>yogurt</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
      <td>Organic 4% Milk Fat Whole Milk Cottage Cheese</td>
      <td>other creams cheeses</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Celery Hearts</td>
      <td>fresh vegetables</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
      <td>Cucumber Kirby</td>
      <td>fresh vegetables</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
      <td>Lightly Smoked Sardines in Olive Oil</td>
      <td>canned meat seafood</td>
      <td>112108</td>
    </tr>
  </tbody>
</table>
</div>




```python
orderp2_df = pd.merge(orderp_df,products_df, how = 'left', on = 'product_id') 
```


```python
orderp2_df = pd.merge(orderp2_df,aisles_df, how = 'left', on = 'aisle_id')
```


```python
orderp2_df = orderp2_df.drop(['aisle_id','department_id'],axis=1) #axis one indicates columns versus rows
```


```python
# adding the user_id from the orders table
orderp2_df = pd.merge(orderp2_df, orders_df[['order_id','user_id']], how = 'left', on = 'order_id')
```


```python
orderp2_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>33120</td>
      <td>1</td>
      <td>1</td>
      <td>Organic Egg Whites</td>
      <td>eggs</td>
      <td>202279</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>28985</td>
      <td>2</td>
      <td>1</td>
      <td>Michigan Organic Kale</td>
      <td>fresh vegetables</td>
      <td>202279</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>9327</td>
      <td>3</td>
      <td>0</td>
      <td>Garlic Powder</td>
      <td>spices seasonings</td>
      <td>202279</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>45918</td>
      <td>4</td>
      <td>1</td>
      <td>Coconut Butter</td>
      <td>oils vinegars</td>
      <td>202279</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>30035</td>
      <td>5</td>
      <td>0</td>
      <td>Natural Sweetener</td>
      <td>baking ingredients</td>
      <td>202279</td>
    </tr>
  </tbody>
</table>
</div>




```python
frames = [ordert_df, orderp2_df] 
```


```python
order_t_p_concat = pd.concat(frames)
```


```python
order_t_p_concat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
      <td>Bulgarian Yogurt</td>
      <td>yogurt</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
      <td>Organic 4% Milk Fat Whole Milk Cottage Cheese</td>
      <td>other creams cheeses</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Celery Hearts</td>
      <td>fresh vegetables</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
      <td>Cucumber Kirby</td>
      <td>fresh vegetables</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
      <td>Lightly Smoked Sardines in Olive Oil</td>
      <td>canned meat seafood</td>
      <td>112108</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary2 = pd.merge(orders_df, order_t_p_concat, how = "inner" , on = 'order_id') #
```


```python
prod_per_user_summary2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id_x</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>soft drinks</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>soy lactosefree</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>popcorn jerky</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>popcorn jerky</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>paper goods</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary2 = prod_per_user_summary2.drop('user_id_y',1)
```


```python
prod_per_user_summary2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id_x</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>soft drinks</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>soy lactosefree</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>popcorn jerky</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>popcorn jerky</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>paper goods</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = prod_per_user_summary2.groupby(['eval_set'])
g[['order_id']].count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
    </tr>
    <tr>
      <th>eval_set</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>prior</th>
      <td>32434489</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0</td>
    </tr>
    <tr>
      <th>train</th>
      <td>1384617</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create csv to use in later notbook
ordert_df.to_csv('ordert_df', sep='\t')
```

## Orders and product table


```python
orders_products_df = pd.merge(orders_df, orderp_df, how = "inner" , on = 'order_id') #
```


```python
orders_products_df = pd.merge(orders_products_df, products_df, how = "left" , on = 'product_id') #
```


```python
orders_products_df = orders_products_df.drop(['aisle_id','department_id'],axis=1)
```


```python
orders_products_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = orders_products_df.groupby(['eval_set'])
g[['order_id']].count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
    </tr>
    <tr>
      <th>eval_set</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>prior</th>
      <td>32434489</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0</td>
    </tr>
    <tr>
      <th>train</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
orders_products_df.info(memory_usage='deep')
orders_products_df['order_number'] = orders_products_df['order_number'].astype('int32')
orders_products_df.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 11 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int64
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    dtypes: category(2), float16(1), int32(3), int64(1), int8(3), uint8(1)
    memory usage: 1.2 GB
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 11 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    dtypes: category(2), float16(1), int32(4), int8(3), uint8(1)
    memory usage: 1.1 GB
    


```python
# remove unnecessary dfs. 
del orderp_df
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-111-7dc4c01ad85a> in <module>()
          1 # remove unnecessary dfs.
    ----> 2 del orderp_df
    

    NameError: name 'orderp_df' is not defined



```python
del depts_df
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-112-421f23fb8a47> in <module>()
    ----> 1 del depts_df
    

    NameError: name 'depts_df' is not defined



```python
del aisles_df
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-113-04f2e0ac7c06> in <module>()
    ----> 1 del aisles_df
    

    NameError: name 'aisles_df' is not defined


## Develop Product Level metrics

### Create row number based on user_id and product_id to determine first and second orders -> used in later for re-order prob and re-order ratio


```python

product_sum_df = orders_products_df.assign(rn=orders_products_df
                                           .sort_values(['user_id','order_number','product_id'])
                                           .groupby(['user_id','product_id'])
                                           .cumcount()+ 1).reset_index()

```


```python
print(type(orders_df))
print(type(orders_products_df))
print(type(product_sum_df))
print((orders_df.shape))
print((orders_products_df.shape))
print((product_sum_df.shape))
print(product_sum_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>
    (3421083, 7)
    (32434489, 11)
    (32434489, 13)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32434489 entries, 0 to 32434488
    Data columns (total 13 columns):
    index                     int64
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int64
    dtypes: category(2), float16(1), int32(4), int64(2), int8(3), uint8(1)
    memory usage: 1.3 GB
    None
    


```python
product_sum_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32434484</th>
      <td>32434484</td>
      <td>2977660</td>
      <td>206209</td>
      <td>prior</td>
      <td>13</td>
      <td>1</td>
      <td>12</td>
      <td>7.0</td>
      <td>14197</td>
      <td>5</td>
      <td>1</td>
      <td>Tomato Paste</td>
      <td>2</td>
    </tr>
    <tr>
      <th>32434485</th>
      <td>32434485</td>
      <td>2977660</td>
      <td>206209</td>
      <td>prior</td>
      <td>13</td>
      <td>1</td>
      <td>12</td>
      <td>7.0</td>
      <td>38730</td>
      <td>6</td>
      <td>0</td>
      <td>Brownie Crunch High Protein Bar</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32434486</th>
      <td>32434486</td>
      <td>2977660</td>
      <td>206209</td>
      <td>prior</td>
      <td>13</td>
      <td>1</td>
      <td>12</td>
      <td>7.0</td>
      <td>31477</td>
      <td>7</td>
      <td>0</td>
      <td>High Protein Bar Chunky Peanut Butter</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32434487</th>
      <td>32434487</td>
      <td>2977660</td>
      <td>206209</td>
      <td>prior</td>
      <td>13</td>
      <td>1</td>
      <td>12</td>
      <td>7.0</td>
      <td>6567</td>
      <td>8</td>
      <td>0</td>
      <td>Chocolate Peanut Butter Protein Bar</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32434488</th>
      <td>32434488</td>
      <td>2977660</td>
      <td>206209</td>
      <td>prior</td>
      <td>13</td>
      <td>1</td>
      <td>12</td>
      <td>7.0</td>
      <td>22920</td>
      <td>9</td>
      <td>0</td>
      <td>Roasted &amp; Salted Shelled Pistachios</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_sum_df = product_sum_df.drop(['index'],1)
```


```python
print(product_sum_df.info(memory_usage='deep'))
product_sum_df['rn'] = product_sum_df['rn'].astype('int16')
print(product_sum_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32434489 entries, 0 to 32434488
    Data columns (total 12 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int64
    dtypes: category(2), float16(1), int32(4), int64(1), int8(3), uint8(1)
    memory usage: 1.1 GB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 32434489 entries, 0 to 32434488
    Data columns (total 12 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    dtypes: category(2), float16(1), int16(1), int32(4), int8(3), uint8(1)
    memory usage: 902.4 MB
    None
    


```python
product_sum_df[product_sum_df['rn']>1].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2398795</td>
      <td>1</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>15.0</td>
      <td>196</td>
      <td>1</td>
      <td>1</td>
      <td>Soda</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2398795</td>
      <td>1</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>15.0</td>
      <td>12427</td>
      <td>3</td>
      <td>1</td>
      <td>Original Beef Jerky</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2398795</td>
      <td>1</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>15.0</td>
      <td>26088</td>
      <td>5</td>
      <td>1</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>473747</td>
      <td>1</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>12</td>
      <td>21.0</td>
      <td>196</td>
      <td>1</td>
      <td>1</td>
      <td>Soda</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>473747</td>
      <td>1</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>12</td>
      <td>21.0</td>
      <td>12427</td>
      <td>2</td>
      <td>1</td>
      <td>Original Beef Jerky</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### Create field to see how many times a product was ordered in the dataset


```python
#total products
product_total = product_sum_df.groupby('product_id')['product_id'].count()
product_total.head()
```




    product_id
    1    1852
    2      90
    3     277
    4     329
    5      15
    Name: product_id, dtype: int64




```python
#
product_total = pd.DataFrame(product_total)
product_total.columns = ['prod_freq_count']
#product_total.columns = ['product_id','prod_freq_count']
product_total = product_total.reset_index()
product_total.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>prod_freq_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1852</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>277</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>329</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_sum_df[product_sum_df['product_id']==196].shape
```




    (35791, 12)




```python
product_sum_df = pd.merge(product_sum_df, product_total , how = 'left', on='product_id')
```


```python
product_sum_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>1</td>
      <td>35791</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>15935</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>1</td>
      <td>6476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>1</td>
      <td>2523</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1</td>
      <td>1214</td>
    </tr>
  </tbody>
</table>
</div>




```python
#add two lines below to reduce memory usage
print(product_sum_df.info(memory_usage='deep'))
#product_sum_df['product_name'] = product_sum_df['product_name'].astype(category)
product_sum_df['prod_freq_count'] = product_sum_df['prod_freq_count'].astype('int32')
print(product_sum_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 13 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    prod_freq_count           int64
    dtypes: category(2), float16(1), int16(1), int32(4), int64(1), int8(3), uint8(1)
    memory usage: 1.4 GB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 13 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    prod_freq_count           int32
    dtypes: category(2), float16(1), int16(1), int32(5), int8(3), uint8(1)
    memory usage: 1.2 GB
    None
    


```python
del product_total
```

### Create field to sum/count how oftern a product was reordered


```python
product_reordered = product_sum_df.groupby('product_id')['reordered'].aggregate("sum").reset_index()
product_reordered.columns = ['product_id','prod_reorder_sum']
```


```python
product_reordered.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>prod_reorder_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49672</th>
      <td>49684</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>49673</th>
      <td>49685</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>49674</th>
      <td>49686</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>49675</th>
      <td>49687</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>49676</th>
      <td>49688</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_sum_df[product_sum_df['product_id']==49684]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2465653</th>
      <td>3235517</td>
      <td>15858</td>
      <td>prior</td>
      <td>4</td>
      <td>6</td>
      <td>10</td>
      <td>1.0</td>
      <td>49684</td>
      <td>2</td>
      <td>0</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>5264019</th>
      <td>2832300</td>
      <td>33465</td>
      <td>prior</td>
      <td>5</td>
      <td>5</td>
      <td>8</td>
      <td>9.0</td>
      <td>49684</td>
      <td>2</td>
      <td>0</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8766519</th>
      <td>561618</td>
      <td>55524</td>
      <td>prior</td>
      <td>3</td>
      <td>0</td>
      <td>15</td>
      <td>5.0</td>
      <td>49684</td>
      <td>9</td>
      <td>0</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>12612401</th>
      <td>907116</td>
      <td>79791</td>
      <td>prior</td>
      <td>9</td>
      <td>6</td>
      <td>14</td>
      <td>30.0</td>
      <td>49684</td>
      <td>2</td>
      <td>0</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>24356592</th>
      <td>2583499</td>
      <td>154576</td>
      <td>prior</td>
      <td>3</td>
      <td>4</td>
      <td>10</td>
      <td>19.0</td>
      <td>49684</td>
      <td>7</td>
      <td>0</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>26594880</th>
      <td>2295026</td>
      <td>169083</td>
      <td>prior</td>
      <td>1</td>
      <td>5</td>
      <td>12</td>
      <td>NaN</td>
      <td>49684</td>
      <td>3</td>
      <td>0</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>26594893</th>
      <td>1367981</td>
      <td>169083</td>
      <td>prior</td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>28.0</td>
      <td>49684</td>
      <td>4</td>
      <td>1</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <th>29622532</th>
      <td>2551882</td>
      <td>188130</td>
      <td>prior</td>
      <td>5</td>
      <td>5</td>
      <td>8</td>
      <td>2.0</td>
      <td>49684</td>
      <td>5</td>
      <td>0</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>29916110</th>
      <td>2933365</td>
      <td>190076</td>
      <td>prior</td>
      <td>11</td>
      <td>2</td>
      <td>16</td>
      <td>1.0</td>
      <td>49684</td>
      <td>5</td>
      <td>0</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_sum_df = pd.merge(product_sum_df, product_reordered , how = 'left', on='product_id')
```


```python
product_sum_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
      <th>prod_reorder_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>1</td>
      <td>35791</td>
      <td>27791.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>15935</td>
      <td>12923.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>1</td>
      <td>6476</td>
      <td>4797.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>1</td>
      <td>2523</td>
      <td>1360.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1</td>
      <td>1214</td>
      <td>536.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = product_sum_df.groupby(['eval_set'])
g[['order_id']].count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
    </tr>
    <tr>
      <th>eval_set</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>prior</th>
      <td>32434489</td>
    </tr>
    <tr>
      <th>test</th>
      <td>0</td>
    </tr>
    <tr>
      <th>train</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#add two lines below to reduce memory usage
print(product_sum_df.info(memory_usage='deep'))
product_sum_df['prod_reorder_sum'] = product_sum_df['prod_reorder_sum'].astype('float32')
#product_sum_df[['prod_freq_count','rn']] = #product_sum_df[['prod_freq_count','rn']].astype(int32)
print(product_sum_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 14 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    prod_freq_count           int32
    prod_reorder_sum          float64
    dtypes: category(2), float16(1), float64(1), int16(1), int32(5), int8(3), uint8(1)
    memory usage: 1.5 GB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 14 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    prod_freq_count           int32
    prod_reorder_sum          float32
    dtypes: category(2), float16(1), float32(1), int16(1), int32(5), int8(3), uint8(1)
    memory usage: 1.4 GB
    None
    


```python
#product_sum_df = product_sum_df.drop(['prod_reorder_sum_x','prod_reorder_sum_y'],1)
```


```python
del product_reordered
```

### Create field to calculate the number of first orders for a product


```python
#aisle_first_orders = sum(aisle_time == 1)
product_first_order = product_sum_df[product_sum_df['rn']==1].groupby('product_id')['rn'].aggregate("sum").reset_index()
product_first_order.columns = ['product_id','product_first_order']
product_first_order.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_first_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>716.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>78.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>182.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_sum_df[(product_sum_df['rn']==1) & (product_sum_df['product_id']==2)].count()
```




    order_id                  78
    user_id                   78
    eval_set                  78
    order_number              78
    order_dow                 78
    order_hour_of_day         78
    days_since_prior_order    75
    product_id                78
    add_to_cart_order         78
    reordered                 78
    product_name              78
    rn                        78
    prod_freq_count           78
    prod_reorder_sum          78
    dtype: int64




```python
product_sum_df = pd.merge(product_sum_df, product_first_order , how = 'left', on='product_id')
```


```python
product_sum_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
      <th>prod_reorder_sum</th>
      <th>product_first_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>1</td>
      <td>35791</td>
      <td>27791.0</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>15935</td>
      <td>12923.0</td>
      <td>3012.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>1</td>
      <td>6476</td>
      <td>4797.0</td>
      <td>1679.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>1</td>
      <td>2523</td>
      <td>1360.0</td>
      <td>1163.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1</td>
      <td>1214</td>
      <td>536.0</td>
      <td>678.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#add two lines below to reduce memory usage
print(product_sum_df.info(memory_usage='deep'))
product_sum_df['product_first_order'] = product_sum_df['product_first_order'].astype('float32')
print(product_sum_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 15 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    prod_freq_count           int32
    prod_reorder_sum          float32
    product_first_order       float64
    dtypes: category(2), float16(1), float32(1), float64(1), int16(1), int32(5), int8(3), uint8(1)
    memory usage: 1.6 GB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 15 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    prod_freq_count           int32
    prod_reorder_sum          float32
    product_first_order       float32
    dtypes: category(2), float16(1), float32(2), int16(1), int32(5), int8(3), uint8(1)
    memory usage: 1.5 GB
    None
    

### Create field to calculate the number of second orders for a product, which will be used to determine ratios and probability of reorders


```python
#aisle_second_orders = sum(aisle_time == 2)
product_second_order = product_sum_df[product_sum_df['rn']==2].groupby('product_id')['rn'].aggregate("count").reset_index()
product_second_order.columns = ['product_id','product_second_order']
product_second_order.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_second_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>276</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>64</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_sum_df[(product_sum_df['rn']==2) & (product_sum_df['product_id']==5)].count()
```




    order_id                  4
    user_id                   4
    eval_set                  4
    order_number              4
    order_dow                 4
    order_hour_of_day         4
    days_since_prior_order    4
    product_id                4
    add_to_cart_order         4
    reordered                 4
    product_name              4
    rn                        4
    prod_freq_count           4
    prod_reorder_sum          4
    product_first_order       4
    dtype: int64




```python
product_sum_df = pd.merge(product_sum_df, product_second_order , how = 'left', on='product_id')
```


```python
product_sum_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
      <th>prod_reorder_sum</th>
      <th>product_first_order</th>
      <th>product_second_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>1</td>
      <td>35791</td>
      <td>27791.0</td>
      <td>8000.0</td>
      <td>4660.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>15935</td>
      <td>12923.0</td>
      <td>3012.0</td>
      <td>1895.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>1</td>
      <td>6476</td>
      <td>4797.0</td>
      <td>1679.0</td>
      <td>889.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>1</td>
      <td>2523</td>
      <td>1360.0</td>
      <td>1163.0</td>
      <td>471.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1</td>
      <td>1214</td>
      <td>536.0</td>
      <td>678.0</td>
      <td>246.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
del product_second_order
```


```python
#add two lines below to reduce memory usage
print(product_sum_df.info(memory_usage='deep'))
product_sum_df['product_second_order'] = product_sum_df['product_second_order'].astype('float32')
print(product_sum_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 16 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    prod_freq_count           int32
    prod_reorder_sum          float32
    product_first_order       float32
    product_second_order      float64
    dtypes: category(2), float16(1), float32(2), float64(1), int16(1), int32(5), int8(3), uint8(1)
    memory usage: 1.7 GB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 16 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    rn                        int16
    prod_freq_count           int32
    prod_reorder_sum          float32
    product_first_order       float32
    product_second_order      float32
    dtypes: category(2), float16(1), float32(3), int16(1), int32(5), int8(3), uint8(1)
    memory usage: 1.6 GB
    None
    

### Product reorder probability


```python
#aisle_sum$aisle_reorder_probability <- aisle_sum$aisle_second_orders / aisle_sum$aisle_first_orders
product_sum_df['product_reorder_probability'] = product_sum_df['product_second_order'] / product_sum_df['product_first_order']
```


```python
product_sum_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
      <th>prod_reorder_sum</th>
      <th>product_first_order</th>
      <th>product_second_order</th>
      <th>product_reorder_probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>1</td>
      <td>35791</td>
      <td>27791.0</td>
      <td>8000.0</td>
      <td>4660.0</td>
      <td>0.582500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>15935</td>
      <td>12923.0</td>
      <td>3012.0</td>
      <td>1895.0</td>
      <td>0.629150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>1</td>
      <td>6476</td>
      <td>4797.0</td>
      <td>1679.0</td>
      <td>889.0</td>
      <td>0.529482</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>1</td>
      <td>2523</td>
      <td>1360.0</td>
      <td>1163.0</td>
      <td>471.0</td>
      <td>0.404987</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1</td>
      <td>1214</td>
      <td>536.0</td>
      <td>678.0</td>
      <td>246.0</td>
      <td>0.362832</td>
    </tr>
  </tbody>
</table>
</div>



### How often is a product reorder times


```python
#aisle_sum$aisle_reorder_times <- 1 + aisle_sum$aisle_reorders / aisle_sum$aisle_first_orders
product_sum_df['product_reorder_times'] = 1 + product_sum_df['prod_reorder_sum'] / product_sum_df['product_first_order']
```


```python
product_sum_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
      <th>prod_reorder_sum</th>
      <th>product_first_order</th>
      <th>product_second_order</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>1</td>
      <td>35791</td>
      <td>27791.0</td>
      <td>8000.0</td>
      <td>4660.0</td>
      <td>0.582500</td>
      <td>4.473875</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>15935</td>
      <td>12923.0</td>
      <td>3012.0</td>
      <td>1895.0</td>
      <td>0.629150</td>
      <td>5.290504</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>1</td>
      <td>6476</td>
      <td>4797.0</td>
      <td>1679.0</td>
      <td>889.0</td>
      <td>0.529482</td>
      <td>3.857058</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>1</td>
      <td>2523</td>
      <td>1360.0</td>
      <td>1163.0</td>
      <td>471.0</td>
      <td>0.404987</td>
      <td>2.169389</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1</td>
      <td>1214</td>
      <td>536.0</td>
      <td>678.0</td>
      <td>246.0</td>
      <td>0.362832</td>
      <td>1.790560</td>
    </tr>
  </tbody>
</table>
</div>



### Product reorder ratio


```python
#aisle_sum$aisle_reorder_ratio <- aisle_sum$aisle_reorders / aisle_sum$aisle_orders
product_sum_df['product_reorder_ratio'] = product_sum_df['prod_reorder_sum'] / product_sum_df['prod_freq_count']
```


```python
product_sum_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
      <th>prod_reorder_sum</th>
      <th>product_first_order</th>
      <th>product_second_order</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>1</td>
      <td>35791</td>
      <td>27791.0</td>
      <td>8000.0</td>
      <td>4660.0</td>
      <td>0.582500</td>
      <td>4.473875</td>
      <td>0.776480</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>15935</td>
      <td>12923.0</td>
      <td>3012.0</td>
      <td>1895.0</td>
      <td>0.629150</td>
      <td>5.290504</td>
      <td>0.810982</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>1</td>
      <td>6476</td>
      <td>4797.0</td>
      <td>1679.0</td>
      <td>889.0</td>
      <td>0.529482</td>
      <td>3.857058</td>
      <td>0.740735</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>1</td>
      <td>2523</td>
      <td>1360.0</td>
      <td>1163.0</td>
      <td>471.0</td>
      <td>0.404987</td>
      <td>2.169389</td>
      <td>0.539041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1</td>
      <td>1214</td>
      <td>536.0</td>
      <td>678.0</td>
      <td>246.0</td>
      <td>0.362832</td>
      <td>1.790560</td>
      <td>0.441516</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_sum_df = product_sum_df.drop(['product_first_order','product_second_order','prod_reorder_sum'],axis=1)
```


```python
product_sum_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>rn</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>1</td>
      <td>35791</td>
      <td>0.582500</td>
      <td>4.473875</td>
      <td>0.776480</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>15935</td>
      <td>0.629150</td>
      <td>5.290504</td>
      <td>0.810982</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>1</td>
      <td>6476</td>
      <td>0.529482</td>
      <td>3.857058</td>
      <td>0.740735</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>1</td>
      <td>2523</td>
      <td>0.404987</td>
      <td>2.169389</td>
      <td>0.539041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1</td>
      <td>1214</td>
      <td>0.362832</td>
      <td>1.790560</td>
      <td>0.441516</td>
    </tr>
  </tbody>
</table>
</div>



# Develop Product specific Summary Table to Join onto orderst table with the user level table


```python
product_summary_df = product_sum_df.drop(['order_id','user_id','order_number','order_dow','order_hour_of_day','days_since_prior_order','add_to_cart_order','reordered','rn'],1)

```


```python
#product_summary_df = product_summary_df.drop(['add_to_cart_order','reordered','rn'],1).drop_duplicates()
product_summary_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>eval_set</th>
      <th>product_id</th>
      <th>product_name</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>prior</td>
      <td>196</td>
      <td>Soda</td>
      <td>35791</td>
      <td>0.582500</td>
      <td>4.473875</td>
      <td>0.776480</td>
    </tr>
    <tr>
      <th>1</th>
      <td>prior</td>
      <td>14084</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>15935</td>
      <td>0.629150</td>
      <td>5.290504</td>
      <td>0.810982</td>
    </tr>
    <tr>
      <th>2</th>
      <td>prior</td>
      <td>12427</td>
      <td>Original Beef Jerky</td>
      <td>6476</td>
      <td>0.529482</td>
      <td>3.857058</td>
      <td>0.740735</td>
    </tr>
    <tr>
      <th>3</th>
      <td>prior</td>
      <td>26088</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>2523</td>
      <td>0.404987</td>
      <td>2.169389</td>
      <td>0.539041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>prior</td>
      <td>26405</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>1214</td>
      <td>0.362832</td>
      <td>1.790560</td>
      <td>0.441516</td>
    </tr>
  </tbody>
</table>
</div>




```python
#add two lines below to reduce memory usage
#print(product_sum_df.info(memory_usage='deep'))
#product_sum_df['product_name'] = product_sum_df['product_name'].astype(category)
#product_sum_df[['prod_freq_count','rn']] = #product_sum_df[['prod_freq_count','rn']].astype(int32)
print(product_sum_df.info(memory_usage='deep'))
print(product_summary_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 16 columns):
    order_id                       int32
    user_id                        int32
    eval_set                       category
    order_number                   int32
    order_dow                      int8
    order_hour_of_day              int8
    days_since_prior_order         float16
    product_id                     int32
    add_to_cart_order              int8
    reordered                      uint8
    product_name                   category
    rn                             int16
    prod_freq_count                int32
    product_reorder_probability    float32
    product_reorder_times          float32
    product_reorder_ratio          float32
    dtypes: category(2), float16(1), float32(3), int16(1), int32(5), int8(3), uint8(1)
    memory usage: 1.6 GB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 7 columns):
    eval_set                       category
    product_id                     int32
    product_name                   category
    prod_freq_count                int32
    product_reorder_probability    float32
    product_reorder_times          float32
    product_reorder_ratio          float32
    dtypes: category(2), float32(3), int32(2)
    memory usage: 1.0 GB
    None
    


## Develop User Level Metrics



### Create user table


```python
del user_df
del us
del user_df_metric
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-158-729dcc263f1a> in <module>()
    ----> 1 del user_df
          2 del us
          3 del user_df_metric
    

    NameError: name 'user_df' is not defined



```python
user_df = orders_df[orders_df['eval_set']=='prior']
```


```python
user_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2398795</td>
      <td>1</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>473747</td>
      <td>1</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>12</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2254736</td>
      <td>1</td>
      <td>prior</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>431534</td>
      <td>1</td>
      <td>prior</td>
      <td>5</td>
      <td>4</td>
      <td>15</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#add two lines below to reduce memory usage
#print(product_sum_df.info(memory_usage='deep'))
#product_sum_df['product_name'] = product_sum_df['product_name'].astype(category)
#product_sum_df[['prod_freq_count','rn']] = #product_sum_df[['prod_freq_count','rn']].astype(int32)
print(user_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3214874 entries, 0 to 3421081
    Data columns (total 7 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int64
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    dtypes: category(1), float16(1), int32(2), int64(1), int8(2)
    memory usage: 88.9 MB
    None
    

### Find the total orders placed by user


```python
grouped_df = user_df.groupby('user_id')['order_number'].max().reset_index()
grouped_df.columns = ['user_id','total_user_order']
```


```python
grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>total_user_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df[(user_df['user_id']==5)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1909121</td>
      <td>5</td>
      <td>prior</td>
      <td>2</td>
      <td>0</td>
      <td>16</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2267326</td>
      <td>5</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>18</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>157374</td>
      <td>5</td>
      <td>prior</td>
      <td>4</td>
      <td>1</td>
      <td>18</td>
      <td>19.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df = pd.merge(user_df, grouped_df , how = 'left', on='user_id')
```


```python
user_df[(user_df['user_id']==5)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>total_user_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>4</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1909121</td>
      <td>5</td>
      <td>prior</td>
      <td>2</td>
      <td>0</td>
      <td>16</td>
      <td>11.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2267326</td>
      <td>5</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>18</td>
      <td>10.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>44</th>
      <td>157374</td>
      <td>5</td>
      <td>prior</td>
      <td>4</td>
      <td>1</td>
      <td>18</td>
      <td>19.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
del grouped_df
```

### How long was the user's order period?


```python
grouped_df = user_df.groupby('user_id')['days_since_prior_order'].agg(['sum','mean']).reset_index()
grouped_df.columns = ['user_id','user_order_period','user_mean_days_since_prior_order']
```


```python
grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>176.0</td>
      <td>19.562500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>198.0</td>
      <td>15.234375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>133.0</td>
      <td>12.093750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>55.0</td>
      <td>13.750000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>40.0</td>
      <td>13.335938</td>
    </tr>
  </tbody>
</table>
</div>




```python
# test
d = user_df[(user_df['user_id']==1)]
d['days_since_prior_order'].sum()
```




    176.0




```python
# test
d = user_df[(user_df['user_id']==1)]
d['days_since_prior_order'].mean()
```




    19.56




```python
user_df = pd.merge(user_df, grouped_df , how = 'left', on='user_id')
```


```python
del grouped_df
```


```python
user_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2398795</td>
      <td>1</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>15.0</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
    </tr>
    <tr>
      <th>2</th>
      <td>473747</td>
      <td>1</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>12</td>
      <td>21.0</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2254736</td>
      <td>1</td>
      <td>prior</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>29.0</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>431534</td>
      <td>1</td>
      <td>prior</td>
      <td>5</td>
      <td>4</td>
      <td>15</td>
      <td>28.0</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
    </tr>
  </tbody>
</table>
</div>




```python
#add two lines below to reduce memory usage
#print(product_sum_df.info(memory_usage='deep'))
#product_sum_df['product_name'] = product_sum_df['product_name'].astype(category)
#product_sum_df[['prod_freq_count','rn']] = #product_sum_df[['prod_freq_count','rn']].astype(int32)
print(user_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3214874 entries, 0 to 3214873
    Data columns (total 10 columns):
    order_id                            int32
    user_id                             int32
    eval_set                            category
    order_number                        int64
    order_dow                           int8
    order_hour_of_day                   int8
    days_since_prior_order              float16
    total_user_order                    int64
    user_order_period                   float16
    user_mean_days_since_prior_order    float16
    dtypes: category(1), float16(3), int32(2), int64(2), int8(2)
    memory usage: 125.7 MB
    None
    

### User and products metrics


```python
us = orders_products_df
print(us.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 11 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    dtypes: category(2), float16(1), int32(4), int8(3), uint8(1)
    memory usage: 1.1 GB
    None
    


```python
grouped_df = us.groupby('user_id')['product_id'].count().reset_index()
grouped_df.columns = ['user_id','user_total_products']
```


```python
grouped_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_total_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>195</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>




```python
#test
us[us['user_id']==1].shape
```




    (59, 11)




```python
us = pd.merge(us, grouped_df , how = 'left', on='user_id')
```


```python
del grouped_df
```


```python
us[(us['user_id']==5)].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>user_total_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>360</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>15349</td>
      <td>1</td>
      <td>0</td>
      <td>Organic Raw Agave Nectar</td>
      <td>37</td>
    </tr>
    <tr>
      <th>361</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>21413</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Soba</td>
      <td>37</td>
    </tr>
    <tr>
      <th>362</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>48775</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Red Cabbage</td>
      <td>37</td>
    </tr>
    <tr>
      <th>363</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>28289</td>
      <td>4</td>
      <td>0</td>
      <td>Organic Shredded Carrots</td>
      <td>37</td>
    </tr>
    <tr>
      <th>364</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>8518</td>
      <td>5</td>
      <td>0</td>
      <td>Organic Red Onion</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>




```python
#add two lines below to reduce memory usage
#print(product_sum_df.info(memory_usage='deep'))
#product_sum_df['product_name'] = product_sum_df['product_name'].astype(category)
#product_sum_df[['prod_freq_count','rn']] = #product_sum_df[['prod_freq_count','rn']].astype(int32)
print(us.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 12 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    user_total_products       int64
    dtypes: category(2), float16(1), int32(4), int64(1), int8(3), uint8(1)
    memory usage: 1.3 GB
    None
    


```python
a = us.groupby('user_id')['reordered'].sum().reset_index()
a.columns = ['user_id','user_reordered_count']
b = us[us['order_number']>1].groupby('user_id')['product_id'].count().reset_index()
b.columns = ['user_id','user_2nd_products']
grouped_df = pd.merge(a,b, how='inner', on='user_id')
grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_reordered_count</th>
      <th>user_2nd_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>41.0</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>93.0</td>
      <td>182</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>55.0</td>
      <td>78</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.0</td>
      <td>14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>14.0</td>
      <td>26</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_df['user_reorder_ratio'] = grouped_df['user_reordered_count'] / grouped_df['user_2nd_products']
grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_reordered_count</th>
      <th>user_2nd_products</th>
      <th>user_reorder_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>41.0</td>
      <td>54</td>
      <td>0.759259</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>93.0</td>
      <td>182</td>
      <td>0.510989</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>55.0</td>
      <td>78</td>
      <td>0.705128</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.0</td>
      <td>14</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>14.0</td>
      <td>26</td>
      <td>0.538462</td>
    </tr>
  </tbody>
</table>
</div>




```python
grouped_df = grouped_df.drop(['user_reordered_count','user_2nd_products'],1)
```


```python
us = pd.merge(us, grouped_df , how = 'inner', on='user_id')
```


```python
us[(us['user_id']==5)].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>360</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>15349</td>
      <td>1</td>
      <td>0</td>
      <td>Organic Raw Agave Nectar</td>
      <td>37</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>361</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>21413</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Soba</td>
      <td>37</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>362</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>48775</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Red Cabbage</td>
      <td>37</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>363</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>28289</td>
      <td>4</td>
      <td>0</td>
      <td>Organic Shredded Carrots</td>
      <td>37</td>
      <td>0.538462</td>
    </tr>
    <tr>
      <th>364</th>
      <td>2717275</td>
      <td>5</td>
      <td>prior</td>
      <td>1</td>
      <td>3</td>
      <td>12</td>
      <td>NaN</td>
      <td>8518</td>
      <td>5</td>
      <td>0</td>
      <td>Organic Red Onion</td>
      <td>37</td>
      <td>0.538462</td>
    </tr>
  </tbody>
</table>
</div>




```python
del grouped_df
```


```python
#add two lines below to reduce memory usage
#print(product_sum_df.info(memory_usage='deep'))
#product_sum_df['product_name'] = product_sum_df['product_name'].astype(category)
#product_sum_df[['prod_freq_count','rn']] = #product_sum_df[['prod_freq_count','rn']].astype(int32)
print(us.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 13 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    user_total_products       int64
    user_reorder_ratio        float64
    dtypes: category(2), float16(1), float64(1), int32(4), int64(1), int8(3), uint8(1)
    memory usage: 1.5 GB
    None
    


```python
#user_distinct_products = product_id.count_distinct()
grouped_df = us.groupby('user_id')['product_id'].nunique().reset_index()
grouped_df.columns = ['user_id','user_distinct_products']
grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_distinct_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>102</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
us = pd.merge(us, grouped_df , how = 'left', on='user_id')
```


```python
us[us['user_id']==2].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>2168274</td>
      <td>2</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>32792</td>
      <td>1</td>
      <td>0</td>
      <td>Chipotle Beef &amp; Pork Realstick</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2168274</td>
      <td>2</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>47766</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Avocado</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2168274</td>
      <td>2</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>20574</td>
      <td>3</td>
      <td>0</td>
      <td>Roasted Turkey</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2168274</td>
      <td>2</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>12000</td>
      <td>4</td>
      <td>0</td>
      <td>Baked Organic Sea Salt Crunchy Pea Snack</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2168274</td>
      <td>2</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>48110</td>
      <td>5</td>
      <td>0</td>
      <td>Thin Stackers Brown Rice Lightly Salted</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
  </tbody>
</table>
</div>




```python
#add two lines below to reduce memory usage
print(us.info(memory_usage='deep'))
us['user_reorder_ratio'] = us['user_reorder_ratio'].astype('float32')
us[['user_total_products','user_distinct_products']] = us[['user_total_products','user_distinct_products']].astype('int32')
print(us.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 14 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    user_total_products       int64
    user_reorder_ratio        float64
    user_distinct_products    int64
    dtypes: category(2), float16(1), float64(1), int32(4), int64(2), int8(3), uint8(1)
    memory usage: 1.8 GB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 14 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    user_total_products       int32
    user_reorder_ratio        float32
    user_distinct_products    int32
    dtypes: category(2), float16(1), float32(1), int32(6), int8(3), uint8(1)
    memory usage: 1.4 GB
    None
    

## Create USER level table to join later on ordert table with product table (product_summary_df) created earlier 


```python
us = us.drop(['eval_set','order_number','order_dow','order_hour_of_day','days_since_prior_order','product_name','product_id','add_to_cart_order','reordered','order_id'],1)
```


```python
us[us['user_id']==2].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>2</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
  </tbody>
</table>
</div>




```python
#us=us.drop(['order_id'],1)
```


```python
us = us.drop_duplicates().reset_index()
```


```python
us.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>user_id</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>59</td>
      <td>0.759259</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59</td>
      <td>2</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>2</th>
      <td>254</td>
      <td>3</td>
      <td>88</td>
      <td>0.705128</td>
      <td>33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>342</td>
      <td>4</td>
      <td>18</td>
      <td>0.071429</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>360</td>
      <td>5</td>
      <td>37</td>
      <td>0.538462</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(us.info(memory_usage='deep'))
us[us['user_id']==2].head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 206209 entries, 0 to 206208
    Data columns (total 5 columns):
    index                     206209 non-null int64
    user_id                   206209 non-null int32
    user_total_products       206209 non-null int32
    user_reorder_ratio        206209 non-null float32
    user_distinct_products    206209 non-null int32
    dtypes: float32(1), int32(3), int64(1)
    memory usage: 4.7 MB
    None
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>user_id</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>59</td>
      <td>2</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df[user_df['user_id']==3].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>1374495</td>
      <td>3</td>
      <td>prior</td>
      <td>1</td>
      <td>1</td>
      <td>14</td>
      <td>NaN</td>
      <td>12</td>
      <td>133.0</td>
      <td>12.09375</td>
    </tr>
    <tr>
      <th>25</th>
      <td>444309</td>
      <td>3</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>19</td>
      <td>9.0</td>
      <td>12</td>
      <td>133.0</td>
      <td>12.09375</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3002854</td>
      <td>3</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>16</td>
      <td>21.0</td>
      <td>12</td>
      <td>133.0</td>
      <td>12.09375</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2037211</td>
      <td>3</td>
      <td>prior</td>
      <td>4</td>
      <td>2</td>
      <td>18</td>
      <td>20.0</td>
      <td>12</td>
      <td>133.0</td>
      <td>12.09375</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2710558</td>
      <td>3</td>
      <td>prior</td>
      <td>5</td>
      <td>0</td>
      <td>17</td>
      <td>12.0</td>
      <td>12</td>
      <td>133.0</td>
      <td>12.09375</td>
    </tr>
  </tbody>
</table>
</div>




```python
#product_sum_df.head()
```


```python
#orders_products_df.head()
```

## Merging the two user tables into one


```python
# keeps crashing....finally worked!!!
user_df = pd.merge(user_df, us, how = 'inner' ,on='user_id')
```


```python
#user_df = user_df.drop(['user_total_products_x','user_reorder_ratio_x','user_distinct_products_x','user_total_products_y','user_reorder_ratio_y','user_distinct_products_y'],1)
```


```python
user_df[user_df['user_id']==2].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>index</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2168274</td>
      <td>2</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1501582</td>
      <td>2</td>
      <td>prior</td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>10.0</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1901567</td>
      <td>2</td>
      <td>prior</td>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>3.0</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>13</th>
      <td>738281</td>
      <td>2</td>
      <td>prior</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>8.0</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1673511</td>
      <td>2</td>
      <td>prior</td>
      <td>5</td>
      <td>3</td>
      <td>11</td>
      <td>8.0</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df['user_avg_basket']=user_df['user_total_products']/user_df['total_user_order']
```


```python
user_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>index</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
      <td>0</td>
      <td>59</td>
      <td>0.759259</td>
      <td>18</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2398795</td>
      <td>1</td>
      <td>prior</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>15.0</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
      <td>0</td>
      <td>59</td>
      <td>0.759259</td>
      <td>18</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>473747</td>
      <td>1</td>
      <td>prior</td>
      <td>3</td>
      <td>3</td>
      <td>12</td>
      <td>21.0</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
      <td>0</td>
      <td>59</td>
      <td>0.759259</td>
      <td>18</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2254736</td>
      <td>1</td>
      <td>prior</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>29.0</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
      <td>0</td>
      <td>59</td>
      <td>0.759259</td>
      <td>18</td>
      <td>5.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>431534</td>
      <td>1</td>
      <td>prior</td>
      <td>5</td>
      <td>4</td>
      <td>15</td>
      <td>28.0</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.5625</td>
      <td>0</td>
      <td>59</td>
      <td>0.759259</td>
      <td>18</td>
      <td>5.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df[user_df['user_id']==2].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>index</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2168274</td>
      <td>2</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>11</td>
      <td>NaN</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1501582</td>
      <td>2</td>
      <td>prior</td>
      <td>2</td>
      <td>5</td>
      <td>10</td>
      <td>10.0</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1901567</td>
      <td>2</td>
      <td>prior</td>
      <td>3</td>
      <td>1</td>
      <td>10</td>
      <td>3.0</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>13</th>
      <td>738281</td>
      <td>2</td>
      <td>prior</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>8.0</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1673511</td>
      <td>2</td>
      <td>prior</td>
      <td>5</td>
      <td>3</td>
      <td>11</td>
      <td>8.0</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>59</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df_metric = user_df.drop(['order_id','eval_set','order_number','order_dow','order_hour_of_day','days_since_prior_order','index'],1)
```


```python
user_df_metric[user_df_metric['user_id']==2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df_metric = user_df_metric.drop_duplicates()
```


```python
user_df_metric[user_df_metric['user_id']==2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df_metric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.562500</td>
      <td>59</td>
      <td>0.759259</td>
      <td>18</td>
      <td>5.900000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>12</td>
      <td>133.0</td>
      <td>12.093750</td>
      <td>88</td>
      <td>0.705128</td>
      <td>33</td>
      <td>7.333333</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4</td>
      <td>5</td>
      <td>55.0</td>
      <td>13.750000</td>
      <td>18</td>
      <td>0.071429</td>
      <td>17</td>
      <td>3.600000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>5</td>
      <td>4</td>
      <td>40.0</td>
      <td>13.335938</td>
      <td>37</td>
      <td>0.538462</td>
      <td>23</td>
      <td>9.250000</td>
    </tr>
  </tbody>
</table>
</div>



# TABLE RECAP


```python
# USER LEVEL
user_df_metric.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>10</td>
      <td>176.0</td>
      <td>19.562500</td>
      <td>59</td>
      <td>0.759259</td>
      <td>18</td>
      <td>5.900000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>14</td>
      <td>198.0</td>
      <td>15.234375</td>
      <td>195</td>
      <td>0.510989</td>
      <td>102</td>
      <td>13.928571</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3</td>
      <td>12</td>
      <td>133.0</td>
      <td>12.093750</td>
      <td>88</td>
      <td>0.705128</td>
      <td>33</td>
      <td>7.333333</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4</td>
      <td>5</td>
      <td>55.0</td>
      <td>13.750000</td>
      <td>18</td>
      <td>0.071429</td>
      <td>17</td>
      <td>3.600000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>5</td>
      <td>4</td>
      <td>40.0</td>
      <td>13.335938</td>
      <td>37</td>
      <td>0.538462</td>
      <td>23</td>
      <td>9.250000</td>
    </tr>
  </tbody>
</table>
</div>




```python
user_df_metric.to_csv('user_df_metric', sep='\t')
```


```python
# PRODUCT LEVEL
product_summary_df = product_summary_df.drop_duplicates()
```


```python
product_summary_df = product_summary_df.drop(['eval_set','product_name'],1)
```


```python
product_summary_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>35791</td>
      <td>0.582500</td>
      <td>4.473875</td>
      <td>0.776480</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14084</td>
      <td>15935</td>
      <td>0.629150</td>
      <td>5.290504</td>
      <td>0.810982</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12427</td>
      <td>6476</td>
      <td>0.529482</td>
      <td>3.857058</td>
      <td>0.740735</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26088</td>
      <td>2523</td>
      <td>0.404987</td>
      <td>2.169389</td>
      <td>0.539041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26405</td>
      <td>1214</td>
      <td>0.362832</td>
      <td>1.790560</td>
      <td>0.441516</td>
    </tr>
  </tbody>
</table>
</div>




```python
product_summary_df.to_csv('product_summary_df', sep='\t')
```

## User Product Metrics and Combining the Datasets


```python
prod_per_user_summary = orders_products_df
```


```python
#orders_products_df['order_number'] = orders_products_df['order_number'].astype('int32')
prod_per_user_summary.info(memory_usage='deep')
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 32434489 entries, 0 to 32434488
    Data columns (total 11 columns):
    order_id                  int32
    user_id                   int32
    eval_set                  category
    order_number              int32
    order_dow                 int8
    order_hour_of_day         int8
    days_since_prior_order    float16
    product_id                int32
    add_to_cart_order         int8
    reordered                 uint8
    product_name              category
    dtypes: category(2), float16(1), int32(4), int8(3), uint8(1)
    memory usage: 1.1 GB
    


```python
prod_per_user_summary.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id_x</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>soft drinks</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>soy lactosefree</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>popcorn jerky</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>popcorn jerky</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>paper goods</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary = prod_per_user_summary.drop(['product_name','aisle'],1)
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-240-ddc958723cdd> in <module>()
    ----> 1 prod_per_user_summary = prod_per_user_summary.drop(['product_name','aisle'],1)
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3692                                            index=index, columns=columns,
       3693                                            level=level, inplace=inplace,
    -> 3694                                            errors=errors)
       3695 
       3696     @rewrite_axis_style_signature('mapper', [('copy', True),
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
       3106         for axis, labels in axes.items():
       3107             if labels is not None:
    -> 3108                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       3109 
       3110         if inplace:
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\generic.py in _drop_axis(self, labels, axis, level, errors)
       3138                 new_axis = axis.drop(labels, level=level, errors=errors)
       3139             else:
    -> 3140                 new_axis = axis.drop(labels, errors=errors)
       3141             dropped = self.reindex(**{axis_name: new_axis})
       3142             try:
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\indexes\base.py in drop(self, labels, errors)
       4385             if errors != 'ignore':
       4386                 raise KeyError(
    -> 4387                     'labels %s not contained in axis' % labels[mask])
       4388             indexer = indexer[~mask]
       4389         return self.delete(indexer)
    

    KeyError: "labels ['aisle'] not contained in axis"



```python
grouped_df = prod_per_user_summary .groupby(['user_id','product_id']).size().reset_index(name='user_product_counts')                                
```


```python
grouped_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>product_id</th>
      <th>user_product_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>196</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10258</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10326</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>12427</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13032</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary = pd.merge(prod_per_user_summary, grouped_df , how = 'left', on=['user_id','product_id'])
```


```python
prod_per_user_summary.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>user_product_counts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary = pd.merge(prod_per_user_summary, user_df_metric[['user_id','total_user_order']] , how = 'left', on='user_id')
```


```python
prod_per_user_summary.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>user_product_counts</th>
      <th>total_user_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>2</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>2</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary['user_prod_order_rate'] = prod_per_user_summary['user_product_counts'] / prod_per_user_summary2['total_user_order']
```


```python
prod_per_user_summary.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>user_id</th>
      <th>eval_set</th>
      <th>order_number</th>
      <th>order_dow</th>
      <th>order_hour_of_day</th>
      <th>days_since_prior_order</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>user_product_counts</th>
      <th>total_user_order</th>
      <th>user_prod_order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>196</td>
      <td>1</td>
      <td>0</td>
      <td>Soda</td>
      <td>10</td>
      <td>10</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>14084</td>
      <td>2</td>
      <td>0</td>
      <td>Organic Unsweetened Vanilla Almond Milk</td>
      <td>1</td>
      <td>10</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>12427</td>
      <td>3</td>
      <td>0</td>
      <td>Original Beef Jerky</td>
      <td>10</td>
      <td>10</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26088</td>
      <td>4</td>
      <td>0</td>
      <td>Aged White Cheddar Popcorn</td>
      <td>2</td>
      <td>10</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2539329</td>
      <td>1</td>
      <td>prior</td>
      <td>1</td>
      <td>2</td>
      <td>8</td>
      <td>NaN</td>
      <td>26405</td>
      <td>5</td>
      <td>0</td>
      <td>XL Pick-A-Size Paper Towel Rolls</td>
      <td>2</td>
      <td>10</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary = prod_per_user_summary.drop(['order_id','order_number','order_dow','order_hour_of_day','days_since_prior_order','add_to_cart_order','reordered','total_user_order'],1)
```


```python
prod_per_user_summary = prod_per_user_summary.drop_duplicates()
```


```python
prod_per_user_summary = prod_per_user_summary.drop(['eval_set','product_name'],1)
```


```python
prod_per_user_summary[prod_per_user_summary['user_id']==2].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>product_id</th>
      <th>user_product_counts</th>
      <th>user_prod_order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>59</th>
      <td>2</td>
      <td>32792</td>
      <td>9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>60</th>
      <td>2</td>
      <td>47766</td>
      <td>4</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2</td>
      <td>20574</td>
      <td>2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2</td>
      <td>12000</td>
      <td>5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2</td>
      <td>48110</td>
      <td>2</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>product_id</th>
      <th>user_product_counts</th>
      <th>user_prod_order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>196</td>
      <td>10</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>14084</td>
      <td>1</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>12427</td>
      <td>10</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>26088</td>
      <td>2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>26405</td>
      <td>2</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
prod_per_user_summary.to_csv('prod_per_user_summary', sep='\t')
```

# CREATE FINAL DATASET


```python
print(product_summary_df.info(memory_usage='deep'))
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 49677 entries, 0 to 32313072
    Data columns (total 5 columns):
    product_id                     49677 non-null int32
    prod_freq_count                49677 non-null int32
    product_reorder_probability    45305 non-null float32
    product_reorder_times          49677 non-null float32
    product_reorder_ratio          49677 non-null float32
    dtypes: float32(3), int32(2)
    memory usage: 1.3 MB
    None
    


```python
#test = product_summary_df.drop_duplicates()
```


```python
#print(test.info(memory_usage='deep'))
```


```python
#test.head()
```


```python
#product_summary_df = product_summary_df.drop_duplicates()
```


```python
ordert_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
      <td>Bulgarian Yogurt</td>
      <td>yogurt</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
      <td>Organic 4% Milk Fat Whole Milk Cottage Cheese</td>
      <td>other creams cheeses</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Celery Hearts</td>
      <td>fresh vegetables</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
      <td>Cucumber Kirby</td>
      <td>fresh vegetables</td>
      <td>112108</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
      <td>Lightly Smoked Sardines in Olive Oil</td>
      <td>canned meat seafood</td>
      <td>112108</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data = pd.merge(ordert_df,product_summary_df,how='left',on='product_id')
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
      <td>Bulgarian Yogurt</td>
      <td>yogurt</td>
      <td>112108</td>
      <td>163.0</td>
      <td>0.435484</td>
      <td>2.629032</td>
      <td>0.619632</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
      <td>Organic 4% Milk Fat Whole Milk Cottage Cheese</td>
      <td>other creams cheeses</td>
      <td>112108</td>
      <td>4472.0</td>
      <td>0.550000</td>
      <td>3.493750</td>
      <td>0.713775</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Celery Hearts</td>
      <td>fresh vegetables</td>
      <td>112108</td>
      <td>23826.0</td>
      <td>0.421169</td>
      <td>2.103284</td>
      <td>0.524553</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
      <td>Cucumber Kirby</td>
      <td>fresh vegetables</td>
      <td>112108</td>
      <td>97315.0</td>
      <td>0.566696</td>
      <td>3.243617</td>
      <td>0.691702</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
      <td>Lightly Smoked Sardines in Olive Oil</td>
      <td>canned meat seafood</td>
      <td>112108</td>
      <td>653.0</td>
      <td>0.334311</td>
      <td>1.914956</td>
      <td>0.477795</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data = pd.merge(train_data,user_df_metric,how='left',on='user_id')
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
      <td>Bulgarian Yogurt</td>
      <td>yogurt</td>
      <td>112108</td>
      <td>163.0</td>
      <td>0.435484</td>
      <td>2.629032</td>
      <td>0.619632</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
      <td>Organic 4% Milk Fat Whole Milk Cottage Cheese</td>
      <td>other creams cheeses</td>
      <td>112108</td>
      <td>4472.0</td>
      <td>0.550000</td>
      <td>3.493750</td>
      <td>0.713775</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Celery Hearts</td>
      <td>fresh vegetables</td>
      <td>112108</td>
      <td>23826.0</td>
      <td>0.421169</td>
      <td>2.103284</td>
      <td>0.524553</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
      <td>Cucumber Kirby</td>
      <td>fresh vegetables</td>
      <td>112108</td>
      <td>97315.0</td>
      <td>0.566696</td>
      <td>3.243617</td>
      <td>0.691702</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
      <td>Lightly Smoked Sardines in Olive Oil</td>
      <td>canned meat seafood</td>
      <td>112108</td>
      <td>653.0</td>
      <td>0.334311</td>
      <td>1.914956</td>
      <td>0.477795</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data = pd.merge(train_data,prod_per_user_summary,how='left',on=['user_id','product_id'])
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
      <th>user_product_counts</th>
      <th>user_prod_order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
      <td>Bulgarian Yogurt</td>
      <td>yogurt</td>
      <td>112108</td>
      <td>163.0</td>
      <td>0.435484</td>
      <td>2.629032</td>
      <td>0.619632</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
      <td>Organic 4% Milk Fat Whole Milk Cottage Cheese</td>
      <td>other creams cheeses</td>
      <td>112108</td>
      <td>4472.0</td>
      <td>0.550000</td>
      <td>3.493750</td>
      <td>0.713775</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
      <td>Organic Celery Hearts</td>
      <td>fresh vegetables</td>
      <td>112108</td>
      <td>23826.0</td>
      <td>0.421169</td>
      <td>2.103284</td>
      <td>0.524553</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
      <td>Cucumber Kirby</td>
      <td>fresh vegetables</td>
      <td>112108</td>
      <td>97315.0</td>
      <td>0.566696</td>
      <td>3.243617</td>
      <td>0.691702</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
      <td>Lightly Smoked Sardines in Olive Oil</td>
      <td>canned meat seafood</td>
      <td>112108</td>
      <td>653.0</td>
      <td>0.334311</td>
      <td>1.914956</td>
      <td>0.477795</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_data.to_csv('insta_train_data', sep='\t')
```

# MODEL


```python
y = train_data['reordered']
y.head()
```




    0    1
    1    1
    2    0
    3    0
    4    1
    Name: reordered, dtype: uint8




```python
x = train_data.drop('reordered',1)
x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>product_name</th>
      <th>aisle</th>
      <th>user_id</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
      <th>user_product_counts</th>
      <th>user_prod_order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>Bulgarian Yogurt</td>
      <td>yogurt</td>
      <td>112108</td>
      <td>163.0</td>
      <td>0.435484</td>
      <td>2.629032</td>
      <td>0.619632</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>Organic 4% Milk Fat Whole Milk Cottage Cheese</td>
      <td>other creams cheeses</td>
      <td>112108</td>
      <td>4472.0</td>
      <td>0.550000</td>
      <td>3.493750</td>
      <td>0.713775</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>Organic Celery Hearts</td>
      <td>fresh vegetables</td>
      <td>112108</td>
      <td>23826.0</td>
      <td>0.421169</td>
      <td>2.103284</td>
      <td>0.524553</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>Cucumber Kirby</td>
      <td>fresh vegetables</td>
      <td>112108</td>
      <td>97315.0</td>
      <td>0.566696</td>
      <td>3.243617</td>
      <td>0.691702</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>Lightly Smoked Sardines in Olive Oil</td>
      <td>canned meat seafood</td>
      <td>112108</td>
      <td>653.0</td>
      <td>0.334311</td>
      <td>1.914956</td>
      <td>0.477795</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
sumNullRws = train_data.isnull().sum()
sumNullRws
```




    order_id                                 0
    product_id                               0
    add_to_cart_order                        0
    reordered                                0
    product_name                             0
    aisle                                    0
    user_id                                  0
    prod_freq_count                          9
    product_reorder_probability           2352
    product_reorder_times                    9
    product_reorder_ratio                    9
    total_user_order                         0
    user_order_period                        0
    user_mean_days_since_prior_order         0
    user_total_products                      0
    user_reorder_ratio                       0
    user_distinct_products                   0
    user_avg_basket                          0
    user_product_counts                 555793
    user_prod_order_rate                555793
    dtype: int64




```python
train_data.shape
```




    (1866122, 20)




```python
dataset = train_data.drop(['product_name','aisle'],1)
```


```python
dataset = dataset.fillna(0)
```


```python
dataset.shape
```




    (1866122, 18)




```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>user_id</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
      <th>total_user_order</th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
      <th>user_product_counts</th>
      <th>user_prod_order_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
      <td>112108</td>
      <td>163.0</td>
      <td>0.435484</td>
      <td>2.629032</td>
      <td>0.619632</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
      <td>112108</td>
      <td>4472.0</td>
      <td>0.550000</td>
      <td>3.493750</td>
      <td>0.713775</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
      <td>112108</td>
      <td>23826.0</td>
      <td>0.421169</td>
      <td>2.103284</td>
      <td>0.524553</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
      <td>112108</td>
      <td>97315.0</td>
      <td>0.566696</td>
      <td>3.243617</td>
      <td>0.691702</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
      <td>112108</td>
      <td>653.0</td>
      <td>0.334311</td>
      <td>1.914956</td>
      <td>0.477795</td>
      <td>3</td>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>0.222222</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.to_csv('insta_dataset', sep='\t')
```


```python
Y = dataset['reordered']
Y = np.ravel(Y)
```


```python
Y
```




    array([1, 1, 0, ..., 1, 1, 1], dtype=uint8)




```python
X = dataset.drop(['reordered','product_id','order_id','user_id'],1)
```


```python
X1 = dataset.drop(['reordered', 'user_product_counts', 'user_prod_order_rate','product_id','order_id','user_id'], axis=1)
```


```python
X2 = dataset.drop(['reordered', 'product_reorder_times','product_reorder_ratio','user_product_counts', 'user_prod_order_rate','product_id','order_id','user_id'], axis=1)
```


```python
X3 = dataset.drop(['reordered', 'product_reorder_times','product_reorder_ratio','user_product_counts', 'user_prod_order_rate','user_total_products','product_id','order_id','user_id'], axis=1)
```


```python
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas.plotting import scatter_matrix
warnings.filterwarnings("ignore")
```


```python
plt.figure(figsize=(10,10)) # new plot
corMat = dataset.drop(['product_id','order_id','user_id'],1).corr(method='pearson')
# plot correlation matrix as a heat map
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("CORELATION MATTRIX USING HEAT MAP")
plt.show()
```


![png](output_225_0.png)



```python
dataset_viz = dataset.drop(['user_product_counts', 'user_prod_order_rate','product_id','order_id','user_id'],1)
```


```python
dataset_viz.iloc[:,0:7].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
      <th>prod_freq_count</th>
      <th>product_reorder_probability</th>
      <th>product_reorder_times</th>
      <th>product_reorder_ratio</th>
      <th>total_user_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>163.0</td>
      <td>0.435484</td>
      <td>2.629032</td>
      <td>0.619632</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>4472.0</td>
      <td>0.550000</td>
      <td>3.493750</td>
      <td>0.713775</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>23826.0</td>
      <td>0.421169</td>
      <td>2.103284</td>
      <td>0.524553</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>97315.0</td>
      <td>0.566696</td>
      <td>3.243617</td>
      <td>0.691702</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>653.0</td>
      <td>0.334311</td>
      <td>1.914956</td>
      <td>0.477795</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset_viz.iloc[:,7:].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_order_period</th>
      <th>user_mean_days_since_prior_order</th>
      <th>user_total_products</th>
      <th>user_reorder_ratio</th>
      <th>user_distinct_products</th>
      <th>user_avg_basket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.0</td>
      <td>11.0</td>
      <td>21</td>
      <td>0.692308</td>
      <td>12</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure()
pd.scatter_matrix(dataset_viz, figsize=(15,15))
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](output_229_1.png)



```python
plt.figure()
pd.scatter_matrix(dataset_viz.iloc[:,7:], figsize=(15,15))
plt.show()
```


    <Figure size 432x288 with 0 Axes>



![png](output_230_1.png)


## Model Cross Validation @ kfold = 5

### X - All features


```python
# prepare models
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART2', DecisionTreeClassifier(max_depth=2, random_state=0)))
models.append(('CART4', DecisionTreeClassifier(max_depth=4, random_state=0)))
models.append(('CART6', DecisionTreeClassifier(max_depth=6, random_state=0)))
#models.append(('NB', GaussianNB()))
#models.append(('RF1', RandomForestClassifier(max_depth=1, random_state=0)))
#models.append(('RF2', RandomForestClassifier(max_depth=2, random_state=0)))
#models.append(('RF3', RandomForestClassifier(max_depth=3, random_state=0)))
#models.append(('RF4', RandomForestClassifier(max_depth=4, random_state=0)))
models.append(('RF5', RandomForestClassifier(max_depth=5, random_state=0)))
models.append(('RF6', RandomForestClassifier(max_depth=6, random_state=0)))
models.append(('ADA', AdaBoostClassifier(random_state=0)))
#models.append(('SVM', SVC()))
```


```python
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=5, random_state=7)
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```

    LR: 0.653238 (0.020154)
    CART4: 1.000000 (0.000000)
    CART6: 1.000000 (0.000000)
    RF5: 1.000000 (0.000000)
    ADA: 1.000000 (0.000000)
    


```python
# boxplot algorithm comparison
fig = plt.figure(figsize=(10,10))
fig.suptitle('5 kFold Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
```


![png](output_235_0.png)


#### X- Model is overfit 

### X1 - Removing 'user_product_counts', 'user_prod_order_rate' from the feature list


```python
# prepare models
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART2', DecisionTreeClassifier(max_depth=2, random_state=0)))
models.append(('CART4', DecisionTreeClassifier(max_depth=4, random_state=0)))
models.append(('CART6', DecisionTreeClassifier(max_depth=6, random_state=0)))
#models.append(('NB', GaussianNB()))
#models.append(('RF1', RandomForestClassifier(max_depth=1, random_state=0)))
#models.append(('RF2', RandomForestClassifier(max_depth=2, random_state=0)))
#models.append(('RF3', RandomForestClassifier(max_depth=3, random_state=0)))
#models.append(('RF4', RandomForestClassifier(max_depth=4, random_state=0)))
models.append(('RF5', RandomForestClassifier(max_depth=5, random_state=0)))
models.append(('RF7', RandomForestClassifier(max_depth=7, random_state=0)))
models.append(('ADA', AdaBoostClassifier(random_state=0)))
#models.append(('SVM', SVC()))
```


```python
x1results = []
x1names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=5, random_state=7)
	cv_results = cross_val_score(model, X1, Y, cv=kfold, scoring=scoring)
	x1results.append(cv_results)
	x1names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```

    LR: 0.763881 (0.001678)
    CART4: 0.762742 (0.001309)
    CART6: 0.774709 (0.001227)
    RF5: 0.772603 (0.001100)
    RF6: 0.778805 (0.001729)
    ADA: 0.781485 (0.001354)
    


```python
# boxplot algorithm comparison
fig = plt.figure(figsize=(8,8))
fig.suptitle('5 kFold Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(x1results)
ax.set_xticklabels(x1names)
plt.show()
```


![png](output_240_0.png)


### X2 - Removing 'product_reorder_times','product_reorder_ratio' from the feature list


```python
# prepare models
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART2', DecisionTreeClassifier(max_depth=2, random_state=0)))
models.append(('CART4', DecisionTreeClassifier(max_depth=4, random_state=0)))
models.append(('CART6', DecisionTreeClassifier(max_depth=6, random_state=0)))
#models.append(('NB', GaussianNB()))
#models.append(('RF1', RandomForestClassifier(max_depth=1, random_state=0)))
#models.append(('RF2', RandomForestClassifier(max_depth=2, random_state=0)))
#models.append(('RF3', RandomForestClassifier(max_depth=3, random_state=0)))
#models.append(('RF4', RandomForestClassifier(max_depth=4, random_state=0)))
models.append(('RF5', RandomForestClassifier(max_depth=5, random_state=0)))
models.append(('RF7', RandomForestClassifier(max_depth=7, random_state=0)))
models.append(('ADA', AdaBoostClassifier(random_state=0)))
#models.append(('SVM', SVC()))
```


```python
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=5, random_state=7)
	cv_results = cross_val_score(model, X2, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```

    LR: 0.763328 (0.001607)
    CART4: 0.762733 (0.001162)
    CART6: 0.774199 (0.001222)
    RF5: 0.773549 (0.001304)
    RF7: 0.781261 (0.001339)
    ADA: 0.781751 (0.001485)
    


```python
# boxplot algorithm comparison
fig = plt.figure(figsize=(8,8))
fig.suptitle('X2 5 kFold Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
```


![png](output_244_0.png)


## X3 - Removed 'reordered', 'product_reorder_times','product_reorder_ratio','user_product_counts', 'user_prod_order_rate','user_total_products' Features


```python
# prepare models
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART2', DecisionTreeClassifier(max_depth=2, random_state=0)))
models.append(('CART4', DecisionTreeClassifier(max_depth=4, random_state=0)))
models.append(('CART6', DecisionTreeClassifier(max_depth=6, random_state=0)))
#models.append(('NB', GaussianNB()))
#models.append(('RF1', RandomForestClassifier(max_depth=1, random_state=0)))
#models.append(('RF2', RandomForestClassifier(max_depth=2, random_state=0)))
#models.append(('RF3', RandomForestClassifier(max_depth=3, random_state=0)))
#models.append(('RF4', RandomForestClassifier(max_depth=4, random_state=0)))
models.append(('RF5', RandomForestClassifier(max_depth=5, random_state=0)))
models.append(('RF7', RandomForestClassifier(max_depth=7, random_state=0)))
models.append(('ADA', AdaBoostClassifier(random_state=0)))
#models.append(('SVM', SVC()))
```


```python
x3results = []
x3names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=5, random_state=7)
	cv_results = cross_val_score(model, X3, Y, cv=kfold, scoring=scoring)
	x3results.append(cv_results)
	x3names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```

    LR: 0.755283 (0.005189)
    CART4: 0.762663 (0.001379)
    CART6: 0.774712 (0.001355)
    RF5: 0.769074 (0.001717)
    RF7: 0.779077 (0.001140)
    ADA: 0.781462 (0.001541)
    


```python
# boxplot algorithm comparison
fig = plt.figure(figsize=(8,8))
fig.suptitle('X3 5 kFold Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(x3results)
ax.set_xticklabels(x3names)
plt.show()
```


![png](output_248_0.png)


#### Selected X3 feature list and the AdaBoost model due to the minimal change in the cross validation, yet we were able to remove 3 additional features. 

## Model Cross Validation @ kfold = 10


```python
# prepare models
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART2', DecisionTreeClassifier(max_depth=2, random_state=0)))
models.append(('CART4', DecisionTreeClassifier(max_depth=4, random_state=0)))
models.append(('CART6', DecisionTreeClassifier(max_depth=6, random_state=0)))
#models.append(('NB', GaussianNB()))
#models.append(('RF1', RandomForestClassifier(max_depth=1, random_state=0)))
#models.append(('RF2', RandomForestClassifier(max_depth=2, random_state=0)))
#models.append(('RF3', RandomForestClassifier(max_depth=3, random_state=0)))
#models.append(('RF4', RandomForestClassifier(max_depth=4, random_state=0)))
models.append(('RF5', RandomForestClassifier(max_depth=5, random_state=0)))
models.append(('RF7', RandomForestClassifier(max_depth=7, random_state=0)))
models.append(('ADA', AdaBoostClassifier(random_state=0)))
#models.append(('SVM', SVC()))
```


```python
# evaluate each model in turn
x1k10_results = []
x1k10_names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, X1, Y, cv=kfold, scoring=scoring)
	x1k10_results.append(cv_results)
	x1k10_names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```

    LR: 0.764635 (0.002679)
    CART4: 0.762933 (0.002725)
    CART6: 0.774640 (0.002620)
    RF5: 0.772699 (0.002508)
    RF7: 0.778860 (0.002323)
    ADA: 0.781474 (0.002259)
    


```python
# boxplot algorithm comparison
fig = plt.figure(figsize=(10,10))
fig.suptitle('X1 10 kFold Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(x1k10_results)
ax.set_xticklabels(x1k10_names)
plt.show()
```


![png](output_253_0.png)



```python
# evaluate each model in turn
x2k10_results = []
x2k10_names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, X2, Y, cv=kfold, scoring=scoring)
	x2k10_results.append(cv_results)
	x2k10_names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```

    LR: 0.763124 (0.002322)
    CART4: 0.762730 (0.002697)
    CART6: 0.774293 (0.002695)
    RF5: 0.773929 (0.002812)
    RF7: 0.781024 (0.002482)
    ADA: 0.781686 (0.002389)
    


```python
# boxplot algorithm comparison
fig = plt.figure(figsize=(10,10))
fig.suptitle('X2 10 kFold Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(x2k10_results)
ax.set_xticklabels(x2k10_names)
plt.show()
```


![png](output_255_0.png)



```python
# evaluate each model in turn
x3k10_results = []
x3k10_names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, X3, Y, cv=kfold, scoring=scoring)
	x3k10_results.append(cv_results)
	x3k10_names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```

    LR: 0.751037 (0.007864)
    CART4: 0.762873 (0.002738)
    CART6: 0.774782 (0.002519)
    RF5: 0.768853 (0.002855)
    RF7: 0.779208 (0.002418)
    ADA: 0.781232 (0.002450)
    


```python
# boxplot algorithm comparison
fig = plt.figure(figsize=(10,10))
fig.suptitle('X3 10 kFold Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(x3k10_results)
ax.set_xticklabels(x3k10_names)
plt.show()
```


![png](output_257_0.png)


## Model Predictions and Score


```python
#add model prediction
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X3, Y, test_size=0.20)  
```


```python
#Implement classifier

AdaBoost = AdaBoostClassifier(random_state=7)  
AdaBoost.fit(X_train, y_train) 
```




    AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
              learning_rate=1.0, n_estimators=50, random_state=7)




```python
y_pred = AdaBoost.predict(X_test)
```


```python
#add confusion matrix
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
```

    [[ 58070  53311]
     [ 28541 233303]]
                 precision    recall  f1-score   support
    
              0       0.67      0.52      0.59    111381
              1       0.81      0.89      0.85    261844
    
    avg / total       0.77      0.78      0.77    373225
    
    


```python
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve,roc_auc_score
```


```python
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(1,figsize=(10,10))
    plt.subplot(211)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Thresholds")
    plt.legend(loc="lower right")
    plt.ylim([0,1])
    
    plt.subplot(212)
    plt.plot(recalls,precisions, alpha=0.2,color='red',lw=5)
    plt.xlim([0.02,0.99])
    plt.ylim([0,1.05])
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
```


```python

y_scores = cross_val_predict(AdaBoost, X_train, y_train, cv=5, method="predict_proba")
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores[:,1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
```


![png](output_265_0.png)



```python
def plot_roc_curve(fpr,tpr,label=None):
    plt.plot(fpr,tpr,'b-',label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.xlabel('True Positive Rate')
```


```python
fpr, tpr, thresholds = roc_curve(y_train,y_scores[:,1])
plot_roc_curve(fpr,tpr)
print('Area Under The Curve: ',roc_auc_score(y_train,y_scores[:,1]))
```

    Area Under The Curve:  0.831465268981045
    


![png](output_267_1.png)


# OLD CODE / NOTES

# OLD CODE / NOTES


```python
#df['Result'] = df['Column A']/df['Column B']
```


```python
#df['C'] = df.groupby(['A','B']).cumcount()+1; df
```


```python
product_sum_df = product_sum_df.drop(['prod_freq_count_x','prod_freq_count_y'],1)
```


```python
#df = orders_products_df.drop_duplicates()
#df.shape
```


```python
#user_df = user_df.drop(['user_mean_days_since_prior_order','user_order_period_y','user_order_period_x'],1)
```


```python
#order_prod_prior_df = order_prod_prior_df.merge(products_df, how = 'left', on = 'product_id')
```


```python
#orders_train_df = orders_df[orders_df['eval_set'] == 'train']
```


```python
#order_prod_prior_df = order_prod_prior_df.merge(depts_df, how = 'left', on = 'department_id')
```


```python
#order_prod_prior_df = order_prod_prior_df.drop(['aisle_id','department_id'],axis=1)
```

### below df is ready to be merged with products


```python
#orders_test_df = orders_df[orders_df['eval_set'] == 'test']
```
