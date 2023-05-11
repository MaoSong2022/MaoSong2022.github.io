---
title: { create icon for .appimage }
date: { 2023-05-11 }
description: 
math: false
img_path: /assets/images/
categories: [Linux]
tags: [tricks]
---

This is the tutorial how to create an icon for an Appimage, suppose the name of software is `example`.
1. download the `example.Appimage` file
2. make the `example.Appimage` file executable
```shell
chmod u+x path/to/example.AppImage
```
3. download the icon by searching `example-icon png` and save the icon
4. move icon image `example-icon png` and `example.AppImage` to `/opt/` directory in case we delete them accidentally.
```shell
sudo mv path/to/example.AppImage /opt/
sudo mv path/to/example_icon.png /opt/
```
5. use `vim` to create the `.desktop` file
```shell
sudo vim /usr/share/applications/example.desktop
```
and adding the following lines:
```
[Desktop Entry]
Name=example
Exec=/opt/example.AppImage
Icon=/opt/example-icon.png
comment=example
Type=Application
Terminal=false
Encoding=UTF-8
```
6. wait for a second and open the applications, we will see the `example_icon`.
