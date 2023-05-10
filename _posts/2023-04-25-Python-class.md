---
title: Python Class confused features
date: 2023-04-25
description: 
math: false
img_path: /assets/images/
categories: [programming_language]
tags: [python]
---

最近遇到的两个关于python的特性，看了python官方文档之后才知道不这样写容易有bug

# mutable v.s. immutable
在区分这两个概念之前，我们需要知道python有以下几种基本类型：
- `number`, 数值类型
- `str`, 字符串类型
- `tuple`, 元组类型
- `list`, 列表类型
- `dict`, 字典类型
- `set`, 集合类型

而这几种基本类型又可以被分类为两种类型（假设`tuple`元素不是可变类型）：
- 可变类型(mutable)： `list`, `dict`, `set`
- 不可变类型(immutable)：`number`, `str`,  `tuple` 

由于python中一切皆对象，当我们传递对象作为参数时，我们实际上传递的是指向对象的地址的指针。
可变类型与不可变的类型的区别就在于，我们能否通过该指针去修改对象的值。不可变类型在赋值时，实际上是拷贝了一份对象，然后将新的对象的地址赋予给新的变量

首先是不可变类型赋值的例子：
```python
a = "abcd"
print(id(a))

b = a
b += "e"
print(id(b))
print(a)
print(b)

# output
# 140390599189616
# 140390600881520
# abcd
# abcde
```
可以看到，由于`a`是一个不可变类型，所以`a`和`b`的地址并不一样，并且修改`b`不会对`a`造成影响。
然后是可变类型赋值的例子：
```python
a = ['a', 'b', 'c']
print(id(a))

b = a
b += ['e']
print(id(b))
print(a)
print(b)

# 140374028510528
# 140374028510528
# ['a', 'b', 'c', 'e']
# ['a', 'b', 'c', 'e']
```
这时，因为`a`是一个可变类型，任何对`b`的修改都将直接对`a`产生影响。


# class variables  v.s. instance variables
instance variable是绑定实例的，每个实例都有其自己的一份copy，其在`__init__`方法中进行初始化；而class variable是所有的对象共享一份。因此，当class variable是一个可变类型时，任何对class variable的改动都将影响这个类的所有实例。
下面我们看class variable和instance variable的定义：
```python
class GroupMember:
	leader = "Job"  # class variable
	activities = []    # mistaken use of class variable

	def __init__(self, name):
		self.name = name

	def add_activity(self, activity):
		self.activities.append(activity)
```
这里我们创建了一个`GroupMember`类，该类记录了这个`GroupMember`类的`leader`名字，由于`leader`是唯一的，因此我们将其创建为一个class variable。我们还创建了一个class variable `activities` **用以记录每个成员的活动情况**， 每个成员拥有`name`属性和`add_activity`方法。
实际上当我们使用这个类的时候，会出现很大的错误，我们会发现随着`GroupMember`实例数量的增加，`activities`占用的内存呈几何级增长，这就是我们前面讲到的原因：任何一个成员调用`add_activity`方法时，都会对所有的`GroupMember`的`activities`进行修改。

我们如果用c++的方式来理解的话就是class variable是静态变量，所有的类共享一份：
```c++
class GroupMember{
	static const string leader = "Job";
	static vector<int> activities;
	string name;
	GroupMember(name): name(name){}
	static add_activity(int activity){
		activities.push(activity);
	}
}
```

> 不要使用可变类型作为类变量，确保类变量不会被错误的修改！
{: .prompt-info }

因此，上面类的定义应该写成：
```python
class GroupMember:
	leader = "Job"  # class variable
	
	def __init__(self, name):
		self.name = name
		self.activities = [] # creates a new empty list for each member

	def add_activity(self, activity):
		self.activities.append(activity)
```

# constructor and default value
这也是个究极难找的bug，其形式如下：
```python
class Group:
	def __init__(self, members=[]):
		self.members = members

	def add_member(self, member):
		self.members.append(member)
```
按理来说，没有太大的问题，我们用一个空的列表来初始化`members`属性，然后后续逐个添加。但事实是，`members`也是一个class variable.
我们用以下代码测试
```python
group1 = Group()
print(group1.members)
group1.add_member("Job")
print(group1.members)

group2 = Group()
print(group2.members)
```

我们的预期输出应该是
```python
[]
['Job']
[]
```
但实际上输出的是
```python
[]
['Job']
['Job']
```

这是因为所有的`Group`类都共享一个`members`实例，我们可以输出两个实例的`members`地址：
```python
print(id(group1.members) == id(group2.members))
# True
```
这验证了我们的想法。


## solution
为了解决这个问题，我们应该避免使用可变类型的参数作为默认值，我们可以有如下替代写法：
```python
class Group:
	def __init__(self, members=None):
		self.members = members if members is not None else []

	def add_member(self, member):
		self.members.append(member)
```
我们用上面的代码进行测试得到：
```python
[]
['job']
[]
False
```
这样就得到了我们想要的效果.

# details
这个特性产生的原因有两个：
1. Function is also an object in python
2. **Default parameter values are evaluated from left to right when the function definition is executed.**

第二点比较难以理解，但是从我的角度来看，其实就是不同的参数会产生不同的函数对象。
因此当运行`group1=Group()`时，我们就定义了一个`Group`类，这个类绑定了一个`members`的列表对象。后续我们在运行`group2=Group()`时，我们实际上产生的是`group1`定义的`Group`类的一个实例，由于我们并没有重新运行`Group`类的定义，因此，`group2`和`group1`绑定的是同一个`members`列表对象。
为了测试，我们现在生成一个新的类：
```python
group3 = Group([])
```
测试
```python
print(group3.members)
print(id(group1.members) == id(group3.members))
# []
# False
```
可以看到，即使我们使用了和默认参数相同的值来生成`Group`类的实例`group3`， 这两个实例仍然是不同的实例，这也就验证了原因的正确性。

# Conclusion
总结起来就是：
1. 不要使用可变类型作为函数，类初始化的默认值。这里的可变类型可以是基本类型，也可以是包含可变类型的复合类型（比如上面定义的`Group`类）
2. 在python里，一切皆对象，特别是函数，一个函数对象在第一次被定义时生成，后续调用相同的函数时使用的都是之前定义过的函数对象。

# references
https://docs.python.org/3.9/reference/compound_stmts.html#function-definitions
https://docs.python.org/3.9/tutorial/classes.html#class-definition-syntax
https://web.archive.org/web/20200221224620id_/http://effbot.org/zone/default-values.htm
