---
title: wine的使用方法
categories:
  - 学编程
tags:
  - 混技能
toc: true
date: 2023-10-28 14:04:21
updated: 2023-10-28 14:04:21
comments: true
---

# wine的使用方法

1.下载

先安装wine，可以使windows上的程序在linux上运行。

```
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine64 wine32
wine --version
```

在官网下载绿色版的mobaxterm，解压进入解压后的文件夹，打开终端利用wine安装

```
wine msiexec /i /installer.msi
```

wine安装的软件需要到home目录下，显示隐藏文件，进入wine文件夹

[![pie1Dw4.png](https://z1.ax1x.com/2023/10/28/pie1Dw4.png)

进入device_c进入program Files(x86)就可以看到下载的windows软件了

[![pie16YR.png](https://z1.ax1x.com/2023/10/28/pie16YR.png)](https://imgse.com/i/pie16YR)

想要打开这个软件需要进入该文件夹，找到exe文件，在终端打开输入

wine xxx.exe

[![pie1W6K.png](https://z1.ax1x.com/2023/10/28/pie1W6K.png)](https://imgse.com/i/pie1W6K)

服务器开启SSH服务，客户机创建SSH会话。
