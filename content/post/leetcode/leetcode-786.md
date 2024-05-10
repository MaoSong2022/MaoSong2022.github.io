---
title: 786. K-th Smallest Prime Fraction
date: 2024-05-10 22:39:54+0800
description: k-th smallest prime fraction in an array
tags: 
    - Array
    - Priority Queue
categories:
    - LeetCode
math: true
---

Given an integer array of size $n$ containing prime integers, it can form $n(n-1)/2$ fractions, we are required to find the $k$-th smallest prime fraction.

# Intuition

We can use a priority queue to store prime integers, then we maintain the priority queue.

# Approach

# Complexity

- Time complexity: iterate the array once and maintain the priority queue.
$$O(n^2\log k)$$

- Space complexity: the size of the priority queue
$$O(k)$$

# Code

```c++
class Solution {
    class Compare {
    public:
        bool operator()(const vector<int>& a, const vector<int>& b){
            return 1.0 * a[0] / a[1] < 1.0 * b[0] / b[1]; // the root is the biggest
        }
    };
public:
    vector<int> kthSmallestPrimeFraction(vector<int>& arr, int k) {
        priority_queue<vector<int>, vector<vector<int>>, Compare> pq;
        int n = arr.size();
        for (int r = 1; r <= k + 1; ++r) {
            for (int i = 0; i < n; ++i) {
                if (i + r >= n) break;
                pq.push(vector<int>{arr[i], arr[i + r]});
                if (pq.size() > k) {
                    pq.pop();
                }
            }
        }
        return pq.top();
    }
};
```

# Reference

- [leetcode](https://leetcode.com/problems/k-th-smallest-prime-fraction/description/)
