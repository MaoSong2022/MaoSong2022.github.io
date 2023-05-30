---
title: C++ inline
date: 2023-05-29
math:  false
mermaid: false
img_path: /assets/images/
categories: [programming]
tags: [C++]
---
# Introduction
根据之前学到的关于`inline`的用法，`inline`的含义是告诉编译器，这个函数很简单，在编译的时候不要进行函数调用，而是将函数体在调用的地方展开（内联展开）
```c++
inline int add (int x, int y){
    return x + y;
}

void func(){
    // do something
    int result = add(a, b);
} 
```
上面的代码就等价于
```c++
inline int add (int x, int y){
    return x + y;
}
void func(){
	// do something
	int result = a + b;
}
```
即编译器直接将函数体复制到对应位置的调用处，避免了函数调用(function call)。

有几个关于`inline`的性质：
1. 在类的声明中定义的函数默认都是inline的
2. inline函数是否内联展开是由编译器决定的



尽管`inline`的逻辑很简单，貌似可以优化整体运行时间。但事实上，`inline`很可能被滥用，一个很简单的函数也可能很复杂（比如函数只有一行用以调用另一个方法），这时`inline`反而会降低代码运行效率。因此，现代编译器有自己的一套判断标准，用以决定那些函数可以被内联展开。这样，`inline`就失去了其存在的意义。


# Improvements
上面说到`inline`关于内联展开的用法目前已经被编译器抛弃。但是不同于`register`被弃用，`inline`的另一用法仍然可以解决C++的一些问题。

我们首先看一个例子：
```c++
// foo.h
int foo(){
  static int count = 0;
  return ++count;
}

// file1.cpp
#include "foo.h"
int func1(){
  return foo();
}

// file2.cpp
#include "foo.h"
int func2(){
  return foo();
}

// main.cpp
#include <iostream>

int func1(), func2();
int main(){
	std::cout << func1() << std::endl;
	std::cout << func2() << std::endl;
}

```
如果我们直接编译运行上面的例子的话，会报错：
```shell
$ g++ main.cpp file1.cpp file2.cpp
/usr/bin/ld: /tmp/ccHCiNUj.o: in function `foo()':
file2.cpp:(.text+0x0): multiple definition of `foo()'; /tmp/cchZjRrj.o:file1.cpp:(.text+0x0): first defined here
collect2: error: ld returned 1 exit status
```

这是因为`file1.cpp`和`file2.cpp`是作为独立的编译单元进行编译的，当编译器第二次遇到`foo()`的定义时（注意`file1.cpp`和`file2.cpp`都包含了`foo.h`这个头文件），就会产生重定义的错误（multiple definition）。

而当我们将`foo()`定义为`inline`之后：
```c++
// foo.h
inline int foo(){
  static int count = 0;
  return ++count;
}
```
再次执行上面的编译命令并运行得到
```shell
$ g++ main.cpp file1.cpp file2.cpp && ./a.out
1
2
```
即代码能够正常运行。

这即是`inline`的第二个作用：对于`non-inline`的函数，整个程序中只允许出现一次定义；而对于`inline`函数，在每一个使用该`inline`函数的翻译单元(translation unit)都应该有它的一次定义。

> 尽管在不同的翻译单元内，可以重复定义一个`inline`函数。但是在一个翻译单元内，`inline`函数的定义还是只能呢个有一个。并且，在不同的翻译单元内，要保证：
> 1. 该`inline`函数是可以找到定义的(reachable)
> 2. 所有该`inline`函数的定义是相同的
> 3. 该 `inline`函数必须被声明为`inline`
> 4. 所有该`inline`函数的地址是相同的
{: .prompt-warning }

在C++中，每一个代码文件就是一个翻译单元，并且C++处理头文件的方式就是将头文件插入到代码文件中`#include`的位置。因此，我们可以将`inline`理解为：允许用户在头文件中定义函数和变量，而不需要将定义和声明拆开。


> `inline`与header guards不同的地方在于，header guards确保的是同一个头文件只会被定义（包含）一次。而`inline`确保一个变量或者函数在多个文件中只会被定义一次。
> `inline`与`extern`的不同之处在于，`extern`表明该函数只是一个声明，定义在其他地方。而`inline`则是允许一个程序中有多次该函数的定义
{: .prompt-info }


# Conclusion
总结起来就是，一般来说类的声明和定义是分开的(`.h`和`.cpp`)，`non-inline`函数（定义）要放到`.cpp`文件中，而`inline`函数（定义）要放到`.h`文件中。

# References
https://www.zhihu.com/question/24185638
https://en.cppreference.com/w/cpp/language/definition