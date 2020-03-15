---
title: "Quick Note: Initialize Python List With Prespecified Size"
layout: post
date: 2019-08-21 01:12:32
math: true
published: true
mathjax: true
pagination: 
    enabled: true
paginate_path: "/Posts/page:num/"
---

I wanted to make a quick note about something I found incredibly helpful the other day.

Lists (or ArrayLists, as new Computer Science students are often taught in their CS 101 courses), are a data strucures that are fundamentally based on arrays, but with additional methods associated with them.  Lists are generally filled with an ```append``` method, that fills indices in this array.  Lists are often useful in the case where the number of intial spots that will be filled is unknown.  

The base arrays are generally associated with a ```size``` or ```length``` parameter, that initializes the array to a certain length.  Under the hood (and generally hidden from the user), however, the ```List``` class also has a ```resize``` method that adds available space to the array when a certain percentage of available indices are occupied, technically allowing the size of the list to grow, and grow, and grow...

Perptually applying ```resize```, however, is slow, especially in the case where you're appending a lot of items.  All of the data currently in the ```List``` object will need to be moved into the new, resized array.

I needed to aggregate a large number (couple thousand) of Pandas DataFrame objects, each saved as a single file, into a single DataFrame.  My first thought was to simply incrementally load and append all incoming DataFrames to a list, and then use ```pandas.concat``` to aggregate them all together.  Appending all of these DataFrames together became incredibly time consuming (at this point, I remembered the ```resize``` issue).

A quick Google search led me to the following solution, allowing me to predefine how large I wanted my list to be:

```python
# For simplicity assume we have 10 items
known_size = 10
initialized_list = [None]*known_size

print(len(initialized_list))
10
```

Neat, huh?  And ridiculously simple.  Now, rather than append, we can do the following:

```python
for j, temp_file in enumerate(list_of_files):
    loaded_file = load_file(temp_file)
    initialized_list[j] = loaded_file
```

Because the **memory has already been pre-allocated**, the ```resize``` method is never accessed, and we save time.  I also found [this blog post](http://zwmiller.com/blogs/python_data_structure_speed.html) with some information about timing with regards to Numpy arrays, lists, and tuples -- the author shows that indexing into a Numpy array is actually slower than indexing into a list.  Numpy arrays are primarilly useful in the case where operations can be vectorized -- then they're the clear winners in terms of speed.