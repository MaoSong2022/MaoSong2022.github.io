---
title: Ctrl-C and Ctrl-Z
date: 2023-05-12
description: 
math: false
img_path: /assets/images/
categories: [Linux]
tags: [os]
---
# Introduction
`ctrl+z` is used for **suspending** a process by sending it the signal `SIGTSTP`, which cannot be intercepted by the program.
`ctrl+c` is used to **kill** a process with the signal `SIGINT`, and can be intercepted by a program so it can clean its self up before exiting, or not exit at all.

If we suspend a process (`ctrl+z`), the terminal will tell us it has been suspended:
```shell
[1]+  Stopped                 yes
```

However, if we kill one (`ctrl+c`),  we won't see confirmation other than being dropped back to a shell prompt.

# Extension
When we suspend a process, we can do fancy things with it. For instance, bring it back the foreground with
```shell
fg
```
and running the command
```shell
bg
```
with a program suspended will allow it to run in the background (the output of the program will still go to the TTY)

If we want to kill a suspended program, we can simply do the command
```shell
kill %1
```
If we have multiple suspended commands, running
```shell
jobs
```
will list them, like this
```shell
[1]-  Stopped                 pianobar
[2]+  Stopped                 yes
```
Using `%#`, where `#` is the job number (the one in square brackets from the `jobs` output) with `bg`, `fg`, or `kill`, can be used to do the action on that job.

# References
https://superuser.com/questions/262942/whats-different-between-ctrlz-and-ctrlc-in-unix-command-line